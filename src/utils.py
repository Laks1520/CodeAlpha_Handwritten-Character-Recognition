import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import os

def visualize_dataset(x_data, y_data, num_samples=25, title="Dataset Samples"):
    """Visualize random samples from dataset"""
    indices = np.random.choice(len(x_data), num_samples, replace=False)
    
    rows = int(np.sqrt(num_samples))
    cols = int(np.ceil(num_samples / rows))
    
    plt.figure(figsize=(12, 12))
    for i, idx in enumerate(indices):
        plt.subplot(rows, cols, i+1)
        plt.imshow(x_data[idx].reshape(28, 28), cmap='gray')
        plt.title(f'Label: {y_data[idx]}')
        plt.axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(f'results/{title.lower().replace(" ", "_")}.png')
    plt.show()

def data_augmentation(x_train, y_train):
    """Apply data augmentation"""
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    
    datagen.fit(x_train)
    return datagen

def save_model_architecture(model, filename='results/model_architecture.png'):
    """Save model architecture as image"""
    os.makedirs('results', exist_ok=True)
    keras.utils.plot_model(
        model,
        to_file=filename,
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',
        dpi=150
    )
    print(f"Model architecture saved to {filename}")

def analyze_misclassifications(model, x_test, y_test, num_samples=10):
    """Analyze and visualize misclassified samples"""
    predictions = np.argmax(model.predict(x_test, verbose=0), axis=1)
    misclassified_indices = np.where(predictions != y_test)[0]
    
    if len(misclassified_indices) == 0:
        print("No misclassifications found!")
        return
    
    sample_indices = np.random.choice(
        misclassified_indices, 
        min(num_samples, len(misclassified_indices)), 
        replace=False
    )
    
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(sample_indices):
        plt.subplot(2, num_samples//2, i+1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f'True: {y_test[idx]}\nPred: {predictions[idx]}', 
                  color='red', fontsize=10)
        plt.axis('off')
    
    plt.suptitle('Misclassified Samples', fontsize=16)
    plt.tight_layout()
    plt.savefig('results/misclassifications.png')
    print(f"Analyzed {len(misclassified_indices)} misclassifications")
    plt.show()
