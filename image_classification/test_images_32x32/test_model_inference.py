"""
Sample inference script for testing the trained model with generated test images.
"""
import numpy as np
import tensorflow as tf
from PIL import Image
import os

def load_model(model_path):
    """Load the trained model."""
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    """Preprocess image for model inference."""
    # Load image
    img = Image.open(image_path).convert('L')  # Ensure greyscale
    img_array = np.array(img)
    
    # Debug: check input range
    print(f"  Raw image range: {img_array.min()} to {img_array.max()}")
    
    # Normalize to [0, 1] range
    img_array = img_array.astype(np.float32) / 255.0
    
    # Debug: check normalized range
    print(f"  Normalized range: {img_array.min():.3f} to {img_array.max():.3f}")
    
    # Add batch and channel dimensions: (1, 32, 32, 1)
    img_array = np.expand_dims(img_array, axis=(0, -1))
    
    return img_array

def test_single_image(model, image_path, expected_output=None):
    """Test model with a single image."""
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array, verbose=0)
    
    print(f"Testing: {os.path.basename(image_path)}")
    print(f"Prediction: {prediction[0]}")
    if expected_output:
        print(f"Expected:   {expected_output}")
        print(f"Correct:    {np.array_equal(np.round(prediction[0]).astype(int), expected_output)}")
    print("-" * 50)
    
    return prediction[0]

if __name__ == "__main__":
    # Update these paths
    model_path = "../src/experiments_outputs/2025_08_01_12_59_58/saved_models/best_augmented_model.h5"
    test_images_dir = "."
    
    # Load model
    model = load_model(model_path)
    print(f"Model loaded from: {model_path}")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print("=" * 60)
    
    # Test with generated images
    for filename in os.listdir(test_images_dir):
        if filename.endswith('.png'):
            image_path = os.path.join(test_images_dir, filename)
            test_single_image(model, image_path)
