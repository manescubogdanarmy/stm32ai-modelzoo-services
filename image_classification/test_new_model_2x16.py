"""
Test the new multi-label trained model with our 2x16 test images.
"""
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os

def load_model_and_test():
    """Load the newly trained model and test it."""
    # Load the best model from multi-label training
    model_path = "experiments_outputs_multilabel/2025_08_01_13_40_07/best_model_multilabel.h5"
    model = tf.keras.models.load_model(model_path)
    
    print("New Multi-Label Model Loaded Successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print("=" * 60)
    
    # Load test manifest
    test_dir = "test_images_2x16"
    manifest_path = os.path.join(test_dir, "test_manifest_2x16.json")
    with open(manifest_path, 'r') as f:
        test_info = json.load(f)
    
    print(f"Testing {len(test_info)} images in 2x16 format")
    print("=" * 60)
    
    class_names = ['high_body_temperature', 'muscle_fatigue', 'emg_fatigue', 'poor_posture', 
                   'high_env_temperature', 'high_stress', 'eye_fatigue', 'normal']
    
    correct_predictions = 0
    total_predictions = len(test_info)
    
    for test_case in test_info:
        print(f"\\nTesting: {test_case['name']}")
        print("-" * 40)
        
        # Load and preprocess image
        img = Image.open(test_case['png_file']).convert('L')
        img_resized = img.resize((32, 32), Image.NEAREST)
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))
        
        # Get model prediction
        prediction = model.predict(img_array, verbose=0)[0]
        predicted_classes = (prediction > 0.5).astype(int)
        
        expected = np.array(test_case['expected_output'])
        
        print(f"  Expected:    {expected}")
        print(f"  Predicted:   {predicted_classes}")
        print(f"  Probabilities: {[f'{p:.3f}' for p in prediction]}")
        
        # Check if prediction matches expected
        is_correct = np.array_equal(predicted_classes, expected)
        if is_correct:
            correct_predictions += 1
            print("  ✓ CORRECT")
        else:
            print("  ✗ INCORRECT")
            
        # Show which classes are predicted
        active_classes = [class_names[i] for i, val in enumerate(predicted_classes) if val == 1]
        expected_classes = [class_names[i] for i, val in enumerate(expected) if val == 1]
        print(f"  Expected classes: {expected_classes}")
        print(f"  Predicted classes: {active_classes}")
    
    print("\\n" + "=" * 60)
    accuracy = correct_predictions / total_predictions
    print(f"Overall Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.2%}")
    
    if accuracy > 0.8:
        print("🎉 Excellent! Model performance is very good!")
    elif accuracy > 0.5:
        print("👍 Good! Model performance is decent.")
    else:
        print("⚠️ Model performance needs improvement.")
    
    return accuracy

if __name__ == "__main__":
    accuracy = load_model_and_test()
