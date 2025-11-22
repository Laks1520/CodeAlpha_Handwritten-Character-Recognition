import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from model import create_mnist_cnn, create_emnist_cnn
import os

def load_mnist_data():
    """Load and preprocess MNIST dataset"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize and reshape
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    return (x_train, y_train), (x_test, y_test)

def train_model(model, x_train, y_train, x_test, y_test, epochs=10, batch_size=128):
    """Train the model"""
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,
        min_lr=1e-7
    )
    
    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    return history

def plot_history(history, save_path='results/training_history.png'):
    """Plot training history"""
    os.makedirs('results', exist_ok=True)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history saved to {save_path}")

if __name__ == "__main__":
    # Load data
    print("Loading MNIST data...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    print(f"Training samples: {len(x_train)}, Test samples: {len(x_test)}")
    
    # Create model
    print("\nCreating model...")
    model = create_mnist_cnn(num_classes=10)
    model.summary()
    
    # Train model
    print("\nTraining model...")
    history = train_model(model, x_train, y_train, x_test, y_test, epochs=15)
    
    # Save model
    os.makedirs('models', exist_ok=True)
    model.save('models/mnist_cnn.h5')
    print("\nModel saved to models/mnist_cnn.h5")
    
    # Plot history
    plot_history(history)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {test_acc*100:.2f}%")
    print(f"Final Test Loss: {test_loss:.4f}")
