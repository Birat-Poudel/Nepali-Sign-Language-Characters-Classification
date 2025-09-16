# Data Processing Guide

This document explains the `tfrecord.py` script used for converting Nepali Sign Language Character images into TensorFlow Record (TFRecord) format.

## Overview
This script converts Nepali Sign Language images into TensorFlow Record (TFRecord) format for efficient machine learning training. TFRecords are TensorFlow's binary storage format that provides fast I/O and reduced memory usage during training.

## Step-by-Step Breakdown

### 1. Imports and Dependencies
```python
import os
import random
import tensorflow as tf

from tqdm import tqdm
from sklearn.model_selection import train_test_split
```
- **os**: File system operations
- **random**: Data shuffling
- **tensorflow**: TFRecord creation and image processing
- **tqdm**: Progress bars for visual feedback
- **sklearn**: Stratified data splitting

### 2. Configuration Constants
```python
IMAGE_SIZE = (256, 256) 
TRAIN_SPLIT = 0.7 
TEST_SPLIT = 0.15 
VAL_SPLIT = 0.15  
DATA_DIR = "data" 
OUTPUT_DIR = "tfrecords"
```
- **IMAGE_SIZE**: All images resized to 256x256 pixels for consistency
- **TRAIN_SPLIT**: 70% of data for training the model
- **TEST_SPLIT**: 15% for final model evaluation
- **VAL_SPLIT**: 15% for hyperparameter tuning and validation
- **DATA_DIR**: Source folder containing organized image data
- **OUTPUT_DIR**: Destination for generated TFRecord files

### 3. Image Loading Function
```python
def load_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, IMAGE_SIZE)
    return image
```
**What it does:**
- **tf.io.read_file()**: Reads raw image bytes from disk
- **tf.image.decode_jpeg()**: Decodes JPEG format, ensures 3 channels (RGB)
- **tf.image.convert_image_dtype()**: Converts to float32 and normalizes to [0,1]
- **tf.image.resize()**: Resizes to standard 256x256 dimensions
- **Returns**: Preprocessed image tensor ready for training

### 4. TFRecord Writing Function
```python
def write_tfrecord(file_paths, labels, output_file, pbar=None):
    with tf.io.TFRecordWriter(output_file) as writer:
        for image_path, label in zip(file_paths, labels):
            image = load_image(image_path).numpy()
            image = (image * 255).astype("uint8")
            label = tf.convert_to_tensor(label, dtype=tf.int64).numpy()
            
            feature = {
                "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.encode_jpeg(image).numpy()])),
                "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=feature))
            writer.write(example.SerializeToString())
```
**What it does:**
- **TFRecordWriter**: Creates binary file writer
- **Load and process**: Calls load_image() for each image
- **Scale back**: Converts [0,1] float back to [0,255] uint8 for storage
- **Create features**: Packages image and label into TensorFlow feature format
- **Serialize**: Converts to binary format and writes to file
- **Progress tracking**: Updates tqdm progress bar

### 5. Data Collection Process
```python
for background_type in ["Plain Background", "Random Background"]:
    background_dir = os.path.join(DATA_DIR, background_type)
    if os.path.exists(background_dir):
        class_names = sorted([d for d in os.listdir(background_dir) if d.isdigit()])
        for class_name in class_names:
            class_dir = os.path.join(background_dir, class_name)
            class_images = [os.path.join(class_dir, img) for img in os.listdir(class_dir) if img.endswith('.jpg')]
            image_paths.extend(class_images)
            labels.extend([int(class_name)] * len(class_images))
```
**What it does:**
- **Iterate backgrounds**: Processes both Plain and Random background folders
- **Find classes**: Identifies numeric folders (0-35) representing sign language characters
- **Collect images**: Gathers all .jpg files from each class folder
- **Assign labels**: Uses folder number as class label (0-35)
- **Combine data**: Merges images from both background types

**Data Structure Expected:**
```
data/
├── Plain Background/
│   ├── 0/ (1000 images of character '0')
│   ├── 1/ (1000 images of character '1')
│   └── ... (up to 35)
└── Random Background/
    ├── 0/ (1000 images of character '0')
    ├── 1/ (1000 images of character '1')
    └── ... (up to 35)
```

### 6. Data Shuffling
```python
combined = list(zip(image_paths, labels))
random.shuffle(combined)
image_paths, labels = zip(*combined)
```
**Why shuffle:**
- Prevents model from learning order-based patterns
- Ensures random distribution across batches
- Improves training stability

### 7. Stratified Data Splitting
```python
train_paths, remaining_paths, train_labels, remaining_labels = train_test_split(
    image_paths, labels, train_size=TRAIN_SPLIT, stratify=labels, random_state=42
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    remaining_paths, remaining_labels, train_size=VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT),
    stratify=remaining_labels, random_state=42
)
```
**What stratified splitting does:**
- **Maintains class balance**: Each split has proportional representation of all 36 classes
- **Two-step process**: First splits train vs (test+val), then splits test vs val
- **Random state**: Ensures reproducible splits across runs
- **Prevents bias**: Avoids scenarios where some classes are missing from splits

### 8. TFRecord File Generation
```python
with tqdm(total=len(train_paths), desc="Training TFRecord", unit="image") as pbar:
    write_tfrecord(train_paths, train_labels, train_tfrecord_path, pbar)
```
**Process for each split:**
- Creates separate .tfrecord files for train, validation, and test
- Shows progress bar with image count
- Processes thousands of images efficiently
- Results in optimized binary files for TensorFlow training

## Output Files
- **train.tfrecord**: ~70% of images for model training
- **val.tfrecord**: ~15% for hyperparameter tuning
- **test.tfrecord**: ~15% for final model evaluation

## Benefits of TFRecord Format
1. **Fast I/O**: Binary format loads faster than individual image files
2. **Memory efficient**: Compressed storage reduces disk space
3. **Batch processing**: Optimized for TensorFlow data pipelines
4. **Preprocessing**: Images already resized and normalized
5. **Reproducible**: Consistent data splits across experiments