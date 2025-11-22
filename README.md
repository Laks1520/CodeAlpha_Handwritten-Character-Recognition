# Handwritten Character Recognition

CNN-based handwritten character recognition system using MNIST and EMNIST datasets.

## Features
- MNIST digit recognition (0-9)
- EMNIST character recognition (letters + digits)
- Deep CNN architecture with dropout
- Real-time prediction on custom images
- Extendable to full word/sentence recognition

### Train Model
```bash
python src/train.py
```

### Make Predictions
```bash
python src/predict.py
```

## Model Architecture
- Convolutional layers with ReLU activation
- Max pooling layers
- Batch normalization
- Dropout for regularization
- Softmax output layer

## Results
- MNIST Test Accuracy: ~99%
- EMNIST Test Accuracy: ~88%

## Dataset
- **MNIST**: 60,000 training + 10,000 test images (digits 0-9)
- **EMNIST**: 697,932 training images (62 classes: digits + uppercase + lowercase)

## Project Structure
```
handwritten-character-recognition/
├── data/                  # Dataset storage
├── models/                # Saved trained models
├── notebooks/             # Jupyter notebooks
├── src/                   # Source code
│   ├── train.py          # Training script
│   ├── model.py          # CNN architecture
│   ├── predict.py        # Prediction script
│   └── utils.py          # Helper functions
├── results/               # Plots and outputs
└── requirements.txt       # Dependencies
```

## Future Enhancements
- Sequence modeling with CRNN for word recognition
- Data augmentation
- Transfer learning
- Web interface for real-time prediction

## License
MIT License
