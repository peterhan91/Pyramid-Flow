# TextVideoDataset

A modern video dataset class that replaces the original `VideoDataset` with enhanced features including text annotations, better video processing, and improved compatibility.

## Features

- **Text Annotations**: Each video/image has associated text descriptions
- **Flexible Input**: Supports both videos and images 
- **Better Processing**: Advanced frame sampling with configurable intervals
- **Smart Cropping**: Center crop to square format with proper resizing
- **Full Compatibility**: Drop-in replacement for original VideoDataset
- **Error Handling**: Robust error handling with fallback sampling

## Data Format

### CSV Metadata File
```csv
file_name,text
video1.mp4,A person walking in the park
video2.mp4,A cat playing with a ball  
image1.jpg,A beautiful sunset
```

### Directory Structure
```
/path/to/video/base/dir/
├── train/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── image1.jpg
└── metadata.csv
```

## Usage

### 1. Basic Usage with VAE Training

```bash
python train/train_video_vae.py \
    --model_arch wan_vae \
    --use_text_video_dataset \
    --text_video_base_path /path/to/video/base/dir \
    --text_video_metadata /path/to/metadata.csv \
    --resolution 256 \
    --max_frames 24 \
    --frame_interval 1 \
    --output_dir ./outputs \
    --lpips_ckpt /path/to/lpips/checkpoint.pth
```

### 2. Python API Usage

```python
from dataset.text_video_dataset import TextVideoDataset

# Create dataset
dataset = TextVideoDataset(
    base_path="/path/to/video/base/dir",
    metadata_path="/path/to/metadata.csv", 
    resolution=256,
    max_frames=24,
    frame_interval=1,
    add_normalize=True
)

# Get item
item = dataset[0]
print(f"Video shape: {item['video'].shape}")  # [C, T, H, W]
print(f"Text: {item['text']}")
print(f"Identifier: {item['identifier']}")  # 'video'
```

### 3. Backward Compatibility

The original VideoDataset still works:
```bash
python train/train_video_vae.py \
    --model_arch wan_vae \
    --video_anno /path/to/video_annotations.jsonl \
    --output_dir ./outputs
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_path` | str | - | Base directory containing video files |
| `metadata_path` | str | - | Path to CSV metadata file |
| `resolution` | int | 256 | Target resolution (square) |
| `max_frames` | int | 24 | Number of frames to extract |
| `frame_interval` | int | 1 | Interval between sampled frames |
| `add_normalize` | bool | True | Normalize to [-1, 1] range |
| `is_i2v` | bool | False | Image-to-video mode (unused in VAE) |

## Output Format

Each dataset item returns a dictionary:
```python
{
    "video": torch.Tensor,    # Shape: [C, T, H, W], dtype: float32
    "identifier": str,        # Always 'video' for compatibility  
    "text": str,             # Text description
    "path": str,             # File path for debugging
}
```

## Video Processing Pipeline

1. **Loading**: Uses imageio for robust video loading
2. **Sampling**: Random start frame + configurable interval sampling
3. **Cropping**: Center crop to square aspect ratio
4. **Resizing**: Resize to target resolution with LANCZOS interpolation
5. **Normalization**: Optional [-1, 1] normalization for VAE training
6. **Format**: Convert from [T, C, H, W] to [C, T, H, W]

## Image Support

- Automatically detects image files (.jpg, .jpeg, .png, .webp, .bmp, .tiff)
- Replicates single frame to create video sequence
- Maintains compatibility with video processing pipeline

## Error Handling

- Graceful handling of corrupted videos
- Automatic fallback to random samples on load failure
- Proper resource cleanup (closes imageio readers)
- Informative error messages

## Performance Considerations

- **Memory Efficient**: Loads frames on-demand
- **Fast I/O**: Uses imageio for optimized video reading
- **Smart Sampling**: Random start frames for training diversity
- **Caching**: No caching to conserve memory

## Comparison with Original VideoDataset

| Feature | TextVideoDataset | Original VideoDataset |
|---------|------------------|----------------------|
| Text annotations | ✅ | ❌ |
| CSV metadata | ✅ | Uses JSONL |
| Image support | ✅ | ❌ |
| Advanced cropping | ✅ | Basic |
| Error handling | ✅ | Basic |
| Frame sampling | Configurable | Fixed |
| Compatibility | Full | - |

## Dependencies

- torch
- torchvision
- pandas
- imageio
- PIL (Pillow)
- einops

## Testing

Run the test script to verify integration:
```bash
python test_text_video_dataset.py
```

## Migration Guide

### From Original VideoDataset

1. **Prepare CSV metadata** from your JSONL annotations
2. **Organize videos** in `/base_path/train/` directory  
3. **Add new arguments** to training command:
   ```bash
   # Old way
   --video_anno /path/to/annotations.jsonl
   
   # New way
   --use_text_video_dataset \
   --text_video_base_path /path/to/base/dir \
   --text_video_metadata /path/to/metadata.csv
   ```

### Converting JSONL to CSV

```python
import pandas as pd
import jsonlines

# Read JSONL
data = []
with jsonlines.open('annotations.jsonl') as reader:
    for item in reader:
        data.append({
            'file_name': item['video'],  # Adjust field names as needed
            'text': item.get('text', 'No description')
        })

# Save as CSV
df = pd.DataFrame(data)
df.to_csv('metadata.csv', index=False)
```

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure all dependencies are installed
2. **File Not Found**: Check base_path and metadata_path are correct
3. **Video Loading**: Verify video files are in `/base_path/train/` directory
4. **Shape Mismatch**: Ensure max_frames matches your model expectations

### Debug Mode

Add debug prints to see what's happening:
```python
dataset = TextVideoDataset(..., verbose=True)  # If implemented
```

## Future Enhancements

- [ ] Multi-resolution support
- [ ] Advanced augmentations
- [ ] Caching options
- [ ] Parallel loading
- [ ] Video format validation
- [ ] Metadata validation 