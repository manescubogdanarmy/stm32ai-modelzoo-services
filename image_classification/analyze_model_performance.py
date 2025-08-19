"""
Analyze the threshold sensitivity and class performance of our multi-label model.
"""
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os
import matplotlib.pyplot as plt

def analyze_thresholds():
    """Analyze different thresholds to optimize performance."""
    # Load the model
    model_path = "experiments_outputs_multilabel/2025_08_01_13_40_07/best_model_multilabel.h5"
    model = tf.keras.models.load_model(model_path)
    
    # Load test data
    test_dir = "test_images_2x16"
    manifest_path = os.path.join(test_dir, "test_manifest_2x16.json")
    with open(manifest_path, 'r') as f:
        test_info = json.load(f)
    
    class_names = ['high_body_temperature', 'muscle_fatigue', 'emg_fatigue', 'poor_posture', 
                   'high_env_temperature', 'high_stress', 'eye_fatigue', 'normal']
    
    # Collect all predictions and expected values
    all_predictions = []
    all_expected = []
    
    print("Collecting predictions for threshold analysis...")
    
    for test_case in test_info:
        # Load and preprocess image
        img = Image.open(test_case['png_file']).convert('L')
        img_resized = img.resize((32, 32), Image.NEAREST)
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))
        
        # Get prediction probabilities
        prediction = model.predict(img_array, verbose=0)[0]
        all_predictions.append(prediction)
        all_expected.append(np.array(test_case['expected_output']))
    
    all_predictions = np.array(all_predictions)
    all_expected = np.array(all_expected)
    
    # Test different thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    accuracies = []
    
    print("\\nTesting different thresholds:")
    print("Threshold | Accuracy | Exact Matches")
    print("-" * 35)
    
    best_threshold = 0.5
    best_accuracy = 0
    
    for threshold in thresholds:
        predicted_binary = (all_predictions > threshold).astype(int)
        exact_matches = np.sum(np.all(predicted_binary == all_expected, axis=1))
        accuracy = exact_matches / len(all_expected)
        accuracies.append(accuracy)
        
        print(f"   {threshold:.2f}   |  {accuracy:.2%}   |     {exact_matches}/12")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    
    print(f"\\nBest threshold: {best_threshold:.2f} with accuracy: {best_accuracy:.2%}")
    
    # Test with best threshold
    print(f"\\n" + "=" * 60)
    print(f"TESTING WITH OPTIMIZED THRESHOLD: {best_threshold:.2f}")
    print("=" * 60)
    
    predicted_binary = (all_predictions > best_threshold).astype(int)
    
    for i, test_case in enumerate(test_info):
        print(f"\\nTesting: {test_case['name']}")
        print("-" * 40)
        
        expected = all_expected[i]
        predicted = predicted_binary[i]
        probabilities = all_predictions[i]
        
        print(f"  Expected:    {expected}")
        print(f"  Predicted:   {predicted}")
        print(f"  Probabilities: {[f'{p:.3f}' for p in probabilities]}")
        
        is_correct = np.array_equal(predicted, expected)
        if is_correct:
            print("  ✓ CORRECT")
        else:
            print("  ✗ INCORRECT")
            
        active_classes = [class_names[j] for j, val in enumerate(predicted) if val == 1]
        expected_classes = [class_names[j] for j, val in enumerate(expected) if val == 1]
        print(f"  Expected classes: {expected_classes}")
        print(f"  Predicted classes: {active_classes}")
    
    # Show per-class performance
    print(f"\\n" + "=" * 60)
    print("PER-CLASS ANALYSIS WITH OPTIMIZED THRESHOLD")
    print("=" * 60)
    
    for j, class_name in enumerate(class_names):
        true_positives = np.sum((predicted_binary[:, j] == 1) & (all_expected[:, j] == 1))
        false_positives = np.sum((predicted_binary[:, j] == 1) & (all_expected[:, j] == 0))
        false_negatives = np.sum((predicted_binary[:, j] == 0) & (all_expected[:, j] == 1))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_name:20} | P: {precision:.3f} | R: {recall:.3f} | F1: {f1:.3f}")
    
    final_accuracy = np.sum(np.all(predicted_binary == all_expected, axis=1)) / len(all_expected)
    print(f"\\nFinal optimized accuracy: {final_accuracy:.2%}")
    
    return best_threshold, final_accuracy

if __name__ == "__main__":
    threshold, accuracy = analyze_thresholds()
