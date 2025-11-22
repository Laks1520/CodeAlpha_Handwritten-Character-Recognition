import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

def load_model(model_path='models/mnist_cnn.h5'):
    """Load trained model"""
    return keras.models.load_model(model_path)

def preprocess_image(image_path):
    """Preprocess custom image for prediction"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1).astype('float32') / 255.0
    return img

def predict_digit(model, image):
    """Predict digit from image"""
    prediction = model.predict(image, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class, confidence

def visualize_predictions(model, x_test, y_test, num_samples=10):
    """Visualize model predictions"""
    os.makedirs('results', exist_ok=True)
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        image = x_test[idx:idx+1]
        true_label = y_test[idx]
        pred_label, confidence = predict_digit(model, image)
        
        plt.subplot(2, num_samples//2, i+1)
        plt.imshow(image.reshape(28, 28), cmap='gray')
        color = 'green' if pred_label == true_label else 'red'
        plt.title(f'True: {true_label}\nPred: {pred_label}\n({confidence:.2%})', 
                  color=color, fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/predictions.png', dpi=150)
    print("Predictions saved to results/predictions.png")
    plt.show()

def plot_confusion_matrix(model, x_test, y_test):
    """Plot confusion matrix"""
    os.makedirs('results', exist_ok=True)
    
    # Get predictions
    y_pred = np.argmax(model.predict(x_test, verbose=0), axis=1)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/confusion_matrix.png', dpi=150)
    print("Confusion matrix saved to results/confusion_matrix.png")
    plt.show()
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    # Load model
    print("Loading model...")
    model = load_model()
    
    # Load test data
    print("Loading test data...")
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0
    
    # Evaluate model
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    
    # Visualize predictions
    print("\nGenerating prediction visualizations...")
    visualize_predictions(model, x_test, y_test, num_samples=10)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(model, x_test, y_test)
