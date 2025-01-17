# Noise-to-Noise Denoising Autoencoder for MNIST

This project implements an improved noise-to-noise denoising autoencoder using TensorFlow/Keras, specifically designed for the MNIST dataset. The model uses a threshold activation function in the final layer to produce binary outputs.

## Features

- Noise-to-noise training methodology
- Custom threshold activation function
- Batch normalization and dropout layers for better training stability
- Automated early stopping and learning rate reduction
- Comprehensive visualization tools
- Support for multiple noise types (Gaussian and salt-and-pepper)

## Requirements

```
tensorflow >= 2.0.0
numpy
matplotlib
```

## Model Architecture

The autoencoder consists of:

### Encoder
- Input Layer (784 neurons)
- Dense Layer (512 neurons) + BatchNorm + Dropout
- Dense Layer (256 neurons) + BatchNorm + Dropout
- Dense Layer (128 neurons) - Encoded Representation

### Decoder
- Dense Layer (256 neurons) + BatchNorm + Dropout
- Dense Layer (512 neurons) + BatchNorm + Dropout
- Dense Layer (784 neurons) with threshold activation

## Usage

1. Basic usage:
```python
from denoising_autoencoder import train_and_evaluate

# Train the model and get results
autoencoder, history, denoised_images = train_and_evaluate()
```

2. Custom noise settings:
```python
from denoising_autoencoder import add_noise

# Generate custom noisy data
noisy_data = add_noise(data, noise_factor=0.3, noise_type='gaussian')
```

3. Save/Load the model:
```python
# Save
autoencoder.save('improved_denoising_autoencoder.h5')

# Load
from tensorflow.keras.models import load_model
loaded_model = load_model('improved_denoising_autoencoder.h5',
                         custom_objects={'threshold_activation': threshold_activation})
```

## Training Details

- Loss Function: Binary Cross-Entropy
- Optimizer: Adam (initial learning rate: 0.001)
- Batch Size: 256
- Maximum Epochs: 200 (with early stopping)
- Validation Split: Using test set for validation

## Callbacks

1. Early Stopping
   - Monitors validation loss
   - Patience: 10 epochs
   - Restores best weights

2. Learning Rate Reduction
   - Monitors validation loss
   - Reduction factor: 0.5
   - Patience: 5 epochs
   - Minimum learning rate: 1e-6

## Visualization

The code provides two types of visualizations:

1. Training History
   - Loss curves (training and validation)
   - MAE curves (training and validation)

2. Image Comparisons
   - Original images
   - Noisy images
   - Denoised results

## Functions

- `load_and_preprocess_data()`: Loads and preprocesses MNIST dataset
- `add_noise()`: Adds Gaussian or salt-and-pepper noise to images
- `create_improved_autoencoder()`: Creates the autoencoder model
- `threshold_activation()`: Custom activation function for binary output
- `plot_training_history()`: Visualizes training metrics
- `plot_results()`: Displays original, noisy, and denoised images
- `train_and_evaluate()`: Main function for training and evaluation

## Results

The model typically achieves:
- Binary output images (black and white)
- Effective noise removal
- Stable training behavior
- Good reconstruction quality