"""
Analyze model predictions to understand what it learned and why predictions are similar.
"""
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import os

def load_model(model_path):
    """Load the trained model."""
    return tf.keras.models.load_model(model_path)

def create_test_patterns():
    """Create distinctive test patterns to see if model can distinguish them."""
    patterns = {}
    
    # Pattern 1: All black
    patterns['all_black'] = np.zeros((32, 32), dtype=np.uint8)
    
    # Pattern 2: All white  
    patterns['all_white'] = np.ones((32, 32), dtype=np.uint8) * 255
    
    # Pattern 3: Checkerboard
    checkerboard = np.zeros((32, 32), dtype=np.uint8)
    for i in range(32):
        for j in range(32):
            if (i + j) % 2 == 0:
                checkerboard[i, j] = 255
    patterns['checkerboard'] = checkerboard
    
    # Pattern 4: Horizontal stripes
    h_stripes = np.zeros((32, 32), dtype=np.uint8)
    for i in range(0, 32, 4):
        h_stripes[i:i+2, :] = 255
    patterns['h_stripes'] = h_stripes
    
    # Pattern 5: Vertical stripes
    v_stripes = np.zeros((32, 32), dtype=np.uint8)
    for j in range(0, 32, 4):
        v_stripes[:, j:j+2] = 255
    patterns['v_stripes'] = v_stripes
    
    # Pattern 6: Center square
    center_square = np.zeros((32, 32), dtype=np.uint8)
    center_square[8:24, 8:24] = 255
    patterns['center_square'] = center_square
    
    return patterns

def test_patterns(model, patterns):
    """Test model with different patterns."""
    print("Testing distinctive patterns:")
    print("=" * 60)
    
    for name, pattern in patterns.items():
        # Normalize and reshape
        normalized = pattern.astype(np.float32) / 255.0
        input_data = np.expand_dims(normalized, axis=(0, -1))
        
        # Get prediction
        prediction = model.predict(input_data, verbose=0)
        pred_values = prediction[0]
        
        # Find highest probability class
        max_class = np.argmax(pred_values)
        max_prob = pred_values[max_class]
        
        print(f"Pattern: {name:15} | Max class: {max_class} ({max_prob:.4f})")
        print(f"Full prediction: {pred_values}")
        print("-" * 60)
        
        # Save pattern for visual inspection
        Image.fromarray(pattern).save(f'pattern_{name}.png')

def analyze_training_distribution():
    """Analyze what the model learned from training data."""
    print("\nAnalyzing model architecture and training data patterns...")
    
    # The sensor classes from user_config.yaml:
    sensor_classes = [
        'all_alert',     # 0
        'sensors_0',     # 1  
        'sensors_1',     # 2
        'sensors_2',     # 3
        'sensors_3',     # 4
        'sensors_4',     # 5
        'sensors_5',     # 6
        'sensors_6'      # 7
    ]
    
    print("Expected output classes:")
    for i, class_name in enumerate(sensor_classes):
        print(f"  Class {i}: {class_name}")
    
    return sensor_classes

def check_model_weights(model):
    """Check if model weights suggest overfitting or other issues."""
    print("\nModel architecture summary:")
    print(f"Total parameters: {model.count_params()}")
    
    # Check if all weights are similar (suggests underfitting)
    all_weights = []
    for layer in model.layers:
        if hasattr(layer, 'get_weights') and layer.get_weights():
            weights = layer.get_weights()[0]  # Get first weight array
            all_weights.extend(weights.flatten())
    
    if all_weights:
        weights_array = np.array(all_weights)
        print(f"Weight statistics:")
        print(f"  Mean: {weights_array.mean():.6f}")
        print(f"  Std:  {weights_array.std():.6f}")
        print(f"  Min:  {weights_array.min():.6f}")
        print(f"  Max:  {weights_array.max():.6f}")
        
        # Check if weights are too uniform (bad sign)
        if weights_array.std() < 0.001:
            print("  WARNING: Very low weight variance - model may not have learned properly!")

if __name__ == "__main__":
    # Load model
    model_path = "../src/experiments_outputs/2025_08_01_12_59_58/saved_models/best_augmented_model.h5"
    model = load_model(model_path)
    
    print("Model Analysis for 32x32 Sensor Classification")
    print("=" * 60)
    
    # Analyze training setup
    sensor_classes = analyze_training_distribution()
    
    # Check model weights
    check_model_weights(model)
    
    # Test with distinctive patterns
    patterns = create_test_patterns()
    test_patterns(model, patterns)
    
    print("\nConclusion:")
    print("If all predictions are very similar across different patterns,")
    print("this suggests the model may have:")
    print("1. Learned to predict the average class distribution")
    print("2. Overfitted to training data (100% validation accuracy is suspicious)")
    print("3. Issues with input preprocessing during training vs inference")
