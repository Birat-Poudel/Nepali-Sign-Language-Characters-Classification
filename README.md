# Nepali Sign Language Classification

A deep learning project for classifying Nepali Sign Language characters using TensorFlow. This project processes sign language images and converts them into TFRecord format for efficient training of neural networks.

## Dataset

**Download Link**: [Nepali Sign Language Character Dataset](https://www.kaggle.com/datasets/biratpoudelrocks/nepali-sign-language-character-dataset)

The dataset contains images of Nepali Sign Language characters (0-35) with two background types:
- **Plain Background**: Clean images with uniform backgrounds
- **Random Background**: Images with varied, realistic backgrounds

For detailed data collection information, see [DATA_COLLECTION.md](DATA_COLLECTION.md)

For detailed data processing information, see [DATA_PROCESSING.md](DATA_PROCESSING.md)

### Dataset Structure
```
data/
├── Plain Background/
│   ├── 0/ (Character '0' images)
│   ├── 1/ (Character '1' images)
│   ├── ...
│   └── 35/ (Character '35' images)
└── Random Background/
    ├── 0/ (Character '0' images)
    ├── 1/ (Character '1' images)
    ├── ...
    └── 35/ (Character '35' images)
```

## Project Structure

```
nsl-classification/
├── data/                    # Raw dataset (Plain & Random Background)
├── tfrecords/               # Processed TFRecord files
│   ├── train.tfrecord       # Training data (70%)
│   ├── val.tfrecord         # Validation data (15%)
│   └── test.tfrecord        # Test data (15%)
├── tfrecord.py              # Data preprocessing script
├── nsl.ipynb                # Main training notebook
├── DATA_PREPARATION.md      # Detailed preprocessing documentation
├── pyproject.toml           # Project dependencies
└── README.md                # README
```

## Features

- **36 Classes**: Nepali Sign Language characters (0-35)
- **Dual Background Types**: Plain and random backgrounds for robustness
- **TFRecord Format**: Optimized binary format for fast training
- **Stratified Splitting**: Balanced train/validation/test splits
- **Image Preprocessing**: Standardized 256x256 pixel images
- **Progress Tracking**: Visual progress bars during data processing

## Requirements

- Python ≥ 3.12
- TensorFlow ≥ 2.20.0
- scikit-learn ≥ 1.7.2
- tqdm ≥ 4.67.1

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd nsl
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```
   
   Or install manually:
   ```bash
   uv add tensorflow scikit-learn tqdm
   ```

## Usage

### 1. Data Preparation

1. **Download the dataset** from [Kaggle](https://www.kaggle.com/datasets/biratpoudelrocks/nepali-sign-language-character-dataset)
2. **Extract** the dataset to the `data/` directory
3. **Run the preprocessing script**:
   ```bash
   python3 tfrecord.py
   ```

This will:
- Process all images from both background types
- Resize images to 256×256 pixels
- Create stratified train/validation/test splits (70%/15%/15%)
- Generate optimized TFRecord files in the `tfrecords/` directory
- Display progress bars for each processing step

### 2. Model Training

Open and run the Jupyter notebook:
```bash
jupyter notebook nsl.ipynb
```

The notebook includes:
- TFRecord loading and parsing
- Data augmentation techniques
- Model architecture definition
- Training loop with validation
- Performance evaluation

## Data Processing Details

### Image Preprocessing
- **Input**: JPEG images of varying sizes
- **Output**: 256×256 RGB images normalized to [0,1]
- **Format**: TFRecord with image and label features

### Data Splits
- **Training**: 70% of data for model training
- **Validation**: 15% for hyperparameter tuning
- **Test**: 15% for final model evaluation
- **Stratification**: Maintains class balance across all splits

### TFRecord Benefits
- **Performance**: 5-10x faster loading compared to individual image files
- **Storage**: Compressed binary format reduces disk usage
- **Memory**: Efficient batch processing for large datasets
- **Reproducibility**: Consistent data splits across experiments

## Model Architecture

The project uses TensorFlow/Keras for building convolutional neural networks suitable for image classification tasks. The notebook explores various architectures optimized for sign language recognition.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is open source. Please check the dataset license on Kaggle for data usage terms.