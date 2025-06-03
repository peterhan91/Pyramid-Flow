import os
import torch
import torchvision
import pandas as pd
import numpy as np
import imageio
import random
from PIL import Image
from torchvision.transforms import v2
from einops import rearrange
from torch.utils.data import Dataset


class TextVideoDataset(Dataset):
    """
    Text-Video dataset that's compatible with VAE training.
    Replaces the original VideoDataset while supporting text annotations.
    """
    def __init__(self, base_path, metadata_path, resolution=256, max_frames=24, 
                 frame_interval=1, add_normalize=True, is_i2v=False):
        """
        Args:
            base_path: Base directory containing video files
            metadata_path: Path to CSV file with columns: file_name, text
            resolution: Target resolution for videos (square)
            max_frames: Number of frames to extract from each video
            frame_interval: Interval between frames when sampling
            add_normalize: Whether to normalize to [-1, 1] range
            is_i2v: Whether this is for image-to-video (not used in VAE training)
        """
        super().__init__()
        
        metadata = pd.read_csv(metadata_path)
        self.paths = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.texts = metadata["text"].to_list()
        
        self.resolution = resolution
        self.max_frames = max_frames
        self.frame_interval = frame_interval
        self.add_normalize = add_normalize
        self.is_i2v = is_i2v
        
        print(f"The training video clip frame number is {max_frames}")
        print(f"Loaded {len(self.paths)} videos from {metadata_path}")
        
        # Frame processing pipeline
        transforms = [
            v2.Resize(size=(resolution, resolution), antialias=True),
            v2.ToTensor(),
        ]
        
        if add_normalize:
            # Normalize to [-1, 1] range (same as original VideoDataset)
            transforms.append(v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))
        
        self.frame_process = v2.Compose(transforms)

    def crop_and_resize(self, image):
        """Center crop and resize image to target resolution"""
        width, height = image.size
        # Crop to square first
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        image = image.crop((left, top, left + min_dim, top + min_dim))
        
        # Resize to target resolution
        image = image.resize((self.resolution, self.resolution), Image.LANCZOS)
        return image

    def load_frames_using_imageio(self, file_path):
        """Load video frames using imageio"""
        try:
            reader = imageio.get_reader(file_path)
            total_frames = reader.count_frames()
            
            # Check if video has enough frames
            required_frames = self.max_frames * self.frame_interval
            if total_frames < required_frames:
                reader.close()
                return None
            
            # Random start frame to add diversity
            max_start = total_frames - required_frames
            start_frame_id = random.randint(0, max_start) if max_start > 0 else 0
            
            frames = []
            for frame_id in range(self.max_frames):
                frame_idx = start_frame_id + frame_id * self.frame_interval
                frame = reader.get_data(frame_idx)
                frame = Image.fromarray(frame).convert("RGB")
                frame = self.crop_and_resize(frame)
                frame = self.frame_process(frame)
                frames.append(frame)
            
            reader.close()
            
            # Stack frames: T x C x H x W -> C x T x H x W (to match VideoDataset format)
            frames = torch.stack(frames, dim=0)  # T x C x H x W
            frames = rearrange(frames, "T C H W -> C T H W")
            
            return frames
            
        except Exception as e:
            print(f"Error loading video {file_path}: {e}")
            return None

    def load_image(self, file_path):
        """Load single image and replicate to create video"""
        try:
            frame = Image.open(file_path).convert("RGB")
            frame = self.crop_and_resize(frame)
            frame = self.frame_process(frame)
            
            # Replicate frame to create video sequence
            frames = frame.unsqueeze(1).repeat(1, self.max_frames, 1, 1)  # C x T x H x W
            
            return frames
            
        except Exception as e:
            print(f"Error loading image {file_path}: {e}")
            return None

    def is_image(self, file_path):
        """Check if file is an image"""
        file_ext = file_path.split(".")[-1].lower()
        return file_ext in ["jpg", "jpeg", "png", "webp", "bmp", "tiff"]

    def __getitem__(self, index):
        """Get item compatible with VAE training script"""
        text = self.texts[index]
        path = self.paths[index]
        
        # Load video or image
        if self.is_image(path):
            video_tensor = self.load_image(path)
        else:
            video_tensor = self.load_frames_using_imageio(path)
        
        # Handle loading failures
        if video_tensor is None:
            print(f'Loading Video Error with {path}')
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        # Ensure correct shape
        assert video_tensor.shape[1] == self.max_frames, f"Expected {self.max_frames} frames, got {video_tensor.shape[1]}"
        
        # Return format compatible with original VideoDataset
        return {
            "video": video_tensor,           # C x T x H x W tensor
            "identifier": 'video',           # Required by training script
            "text": text,                    # Additional text annotation
            "path": path,                    # File path for debugging
        }

    def __len__(self):
        return len(self.paths)


# Backward compatibility: alias for easy replacement
VideoDatasetWithText = TextVideoDataset 