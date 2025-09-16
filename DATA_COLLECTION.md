# Data Collection Guide

This document explains the `data_collect.py` script used for capturing Nepali Sign Language Character images via webcam.

## Overview

The data collection script provides a real-time interface for capturing and preprocessing sign language images. It uses OpenCV for computer vision operations and includes interactive parameter tuning for optimal image quality.

## Script Components

### Directory Setup
```python
train_directory = "data/total"
characters = ['0', '1', '2', '3', '4', '5', '6', '7', '8']
```
- Creates `data/total/` directory structure
- Auto-creates subdirectories for each character

### Camera Configuration
- Uses default camera (index 0)
- Captures 1000 images per character class
- Processes every 5th frame to avoid redundancy
- ROI dimensions: 300×300 pixels at position (270:570, 50:350)

### Image Processing Pipeline

1. **Frame Capture & ROI Extraction**
   - Flips frame horizontally for mirror effect
   - Extracts Region of Interest (ROI) from center
   - Converts to grayscale

2. **Noise Reduction**
   - **Gaussian Blur**: Reduces high-frequency noise
   - **Bilateral Filter**: Preserves edges while smoothing

3. **Thresholding**
   - **Adaptive Threshold**: Handles varying lighting conditions
   - **OTSU Threshold**: Automatic threshold selection for binary conversion

### Interactive Parameters

The script provides real-time parameter adjustment via trackbars:

| Parameter | Range | Purpose |
|-----------|-------|---------|
| Gaussian Kernel | 1-20 | Blur intensity |
| Bilateral Filter d | 0-20 | Neighborhood diameter |
| Bilateral Filter sigmaColor | 0-150 | Color similarity threshold |
| Bilateral Filter sigmaSpace | 0-150 | Spatial similarity threshold |
| Adaptive Threshold Block Size | 3-50 | Local threshold area |
| Adaptive Threshold C | 0-20 | Threshold adjustment constant |

## Usage Instructions

### 1. Setup
```bash
python3 data_collect.py
```

### 2. Interface
- **Main Window**: Shows live camera feed with ROI rectangle
- **ROI Window**: Displays processed binary image
- **Parameters Window**: Contains adjustment trackbars

### 3. Data Capture
- Press character keys (0-8) to start capturing for that class
- Script automatically saves 1000 images per character
- Images saved as: `data/total/{character}/{count}.jpg`

### 4. Controls
- **ESC**: Exit application
- **Character Keys (0-8)**: Start capture for specific character
- **Trackbars**: Adjust preprocessing parameters in real-time

## Output Structure
```
data/total/
├── 0/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
├── 1/
│   ├── 0.jpg
│   ├── 1.jpg
│   └── ...
└── ...
```

## Technical Details

### Image Specifications
- **Input**: 640×480 webcam frames
- **ROI**: 300×300 pixels
- **Output**: Binary images (0-255 values)
- **Format**: JPEG compression

### Processing Optimizations
- Frame skipping (every 5th frame) for efficiency
- Real-time parameter adjustment without restart
- Automatic file counting and naming
- Memory-efficient processing pipeline

## Best Practices

### Lighting Conditions
- Use consistent, diffused lighting
- Avoid harsh shadows or backlighting
- Adjust bilateral filter parameters for lighting changes

### Hand Positioning
- Keep hand within the ROI rectangle
- Maintain consistent distance from camera
- Ensure clear hand visibility against background

### Data Quality
- Capture varied hand orientations
- Include slight position variations
- Monitor processed ROI window for quality
- Adjust thresholding parameters as needed

## Troubleshooting

### Common Issues
- **Camera not detected**: Check camera index (change from 0 to 1)
- **Poor image quality**: Adjust trackbar parameters
- **ROI too dark/bright**: Modify adaptive threshold settings
- **Blurry images**: Reduce Gaussian kernel size

### Parameter Tuning Tips
- Start with default values
- Adjust one parameter at a time
- Monitor ROI window for immediate feedback
- Save optimal settings for consistent results

## Integration

This script generates raw data that can be processed by:
- `tfrecord.py`: Convert to TFRecord format
- `nsl.ipynb`: Model training and evaluation
- Custom preprocessing pipelines

The collected images serve as the foundation for training robust sign language recognition models.