"""
Test the trained model with 2x16 format images that match the training data format.
"""
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os

def load_model(model_path):
    """Load the trained model."""
    return tf.keras.models.load_model(model_path)

def preprocess_2x16_image(image_path):
    """
    Preprocess 2x16 image for model inference.
    The model expects (32, 32, 1) but was trained with 2x16 images that were resized.
    """
    # Load the 2x16 image
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    
    print(f"    Original 2x16 shape: {img_array.shape}")
    print(f"    Original values: {img_array.flatten()}")
    
    # Resize to 32x32 using the same method as training (nearest neighbor)
    img_resized = img.resize((32, 32), Image.NEAREST)
    img_array_32x32 = np.array(img_resized)
    
    print(f"    Resized to 32x32, shape: {img_array_32x32.shape}")
    print(f"    Resized range: {img_array_32x32.min()} to {img_array_32x32.max()}")
    
    # Normalize to [0, 1] range (same as training preprocessing)
    img_normalized = img_array_32x32.astype(np.float32) / 255.0
    
    # Add batch and channel dimensions: (1, 32, 32, 1)
    img_final = np.expand_dims(img_normalized, axis=(0, -1))
    
    return img_final

def test_2x16_images(model, test_dir="test_images_2x16"):
    """Test model with 2x16 format images."""
    # Load test manifest
    manifest_path = os.path.join(test_dir, "test_manifest_2x16.json")
    with open(manifest_path, 'r') as f:
        test_info = json.load(f)
    
    print(f"Testing {len(test_info)} images in 2x16 format")
    print("=" * 80)
    
    class_names = ['high_body_temperature', 'muscle_fatigue', 'emg_fatigue', 'poor_posture', 
                   'high_env_temperature', 'high_stress', 'eye_fatigue', 'normal']
    
    correct_predictions = 0
    total_predictions = len(test_info)
    
    for test_case in test_info:
        print(f"\\nTesting: {test_case['name']}")
        print("-" * 50)
        
        # Load and preprocess image
        img_processed = preprocess_2x16_image(test_case['png_file'])
        
        # Get model prediction
        prediction = model.predict(img_processed, verbose=0)[0]
        predicted_classes = (prediction > 0.5).astype(int)
        
        expected = np.array(test_case['expected_output'])
        
        print(f"    Expected:  {expected}")
        print(f"    Predicted probabilities: {prediction}")
        print(f"    Predicted classes (>0.5): {predicted_classes}")
        
        # Check if prediction matches expected (exact match for multi-label)
        is_correct = np.array_equal(predicted_classes, expected)
        if is_correct:
            correct_predictions += 1
            print("    ✓ CORRECT")
        else:
            print("    ✗ INCORRECT")
            
        # Show which classes are predicted
        active_classes = [class_names[i] for i, val in enumerate(predicted_classes) if val == 1]
        expected_classes = [class_names[i] for i, val in enumerate(expected) if val == 1]
        print(f"    Expected classes: {expected_classes}")
        print(f"    Predicted classes: {active_classes}")
    
    print("\\n" + "=" * 80)
    accuracy = correct_predictions / total_predictions
    print(f"Overall Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.2%}")
    
    if accuracy > 0.8:
        print("✓ Model performance looks good!")
    elif accuracy > 0.5:
        print("⚠ Model performance is moderate. May need retraining.")
    else:
        print("✗ Model performance is poor. Likely needs retraining or debugging.")
    
    return accuracy

if __name__ == "__main__":
    # Load model
    model_path = "src/experiments_outputs/2025_08_01_12_59_58/saved_models/best_augmented_model.h5"
    model = load_model(model_path)
    
    print("Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    # Test with 2x16 images
    accuracy = test_2x16_images(model)
