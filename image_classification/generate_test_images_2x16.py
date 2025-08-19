"""
Generate test images in the correct 2x16 format that matches the training data.
"""
import numpy as np
from PIL import Image
import os
import argparse

def create_sensor_patterns_2x16():
    """Create sensor alert patterns in the original 2x16 format."""
    patterns = {}
    
    # Based on the training data inspection, each sensor seems to have a specific position
    # Let's create patterns that match the training logic
    
    # high_body_temperature: sensor 0 (position 0-2)
    patterns['high_body_temperature'] = np.zeros((2, 16), dtype=np.uint8)
    patterns['high_body_temperature'][0, 0:3] = 255
    
    # muscle_fatigue: sensor 1 (position 2-4) 
    patterns['muscle_fatigue'] = np.zeros((2, 16), dtype=np.uint8)
    patterns['muscle_fatigue'][0, 2:5] = 255
    
    # emg_fatigue: sensor 2 (position 4-6)
    patterns['emg_fatigue'] = np.zeros((2, 16), dtype=np.uint8) 
    patterns['emg_fatigue'][0, 4:7] = 255
    
    # poor_posture: sensor 3 (position 6-8)
    patterns['poor_posture'] = np.zeros((2, 16), dtype=np.uint8)
    patterns['poor_posture'][0, 6:9] = 255
    
    # high_env_temperature: sensor 4 (position 8-9)
    patterns['high_env_temperature'] = np.zeros((2, 16), dtype=np.uint8)
    patterns['high_env_temperature'][0, 8:10] = 255
    
    # high_stress: sensor 5 (position 10-11)
    patterns['high_stress'] = np.zeros((2, 16), dtype=np.uint8)
    patterns['high_stress'][0, 10:12] = 255
    
    # eye_fatigue: sensor 6 (position 12-13)
    patterns['eye_fatigue'] = np.zeros((2, 16), dtype=np.uint8)
    patterns['eye_fatigue'][0, 12:14] = 255
    
    # normal: all sensors off
    patterns['normal'] = np.zeros((2, 16), dtype=np.uint8)
    
    return patterns

def create_combination_patterns():
    """Create combination patterns like the test generator."""
    base_patterns = create_sensor_patterns_2x16()
    combinations = {}
    
    # Multiple sensor alerts
    combinations['high_body_temperature_muscle_fatigue'] = np.zeros((2, 16), dtype=np.uint8)
    combinations['high_body_temperature_muscle_fatigue'][0, 0:3] = 255  # sensor 0
    combinations['high_body_temperature_muscle_fatigue'][0, 2:5] = 255  # sensor 1
    
    combinations['emg_fatigue_poor_posture'] = np.zeros((2, 16), dtype=np.uint8)
    combinations['emg_fatigue_poor_posture'][0, 4:7] = 255  # sensor 2
    combinations['emg_fatigue_poor_posture'][0, 6:9] = 255  # sensor 3
    
    combinations['high_stress_eye_fatigue'] = np.zeros((2, 16), dtype=np.uint8)
    combinations['high_stress_eye_fatigue'][0, 10:12] = 255  # sensor 5
    combinations['high_stress_eye_fatigue'][0, 12:14] = 255  # sensor 6
    
    # All alerts
    combinations['all_alerts'] = np.zeros((2, 16), dtype=np.uint8)
    combinations['all_alerts'][0, 0:14] = 255  # All sensors active
    
    return combinations

def generate_test_images_2x16(output_dir="test_images_2x16"):
    """Generate test images in 2x16 format."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create individual sensor patterns
    individual_patterns = create_sensor_patterns_2x16()
    combination_patterns = create_combination_patterns()
    
    all_patterns = {**individual_patterns, **combination_patterns}
    
    print(f"Generating {len(all_patterns)} test images in 2x16 format...")
    
    test_info = []
    
    for name, pattern in all_patterns.items():
        # Save as PNG
        png_path = os.path.join(output_dir, f"test_2x16_{name}.png")
        img = Image.fromarray(pattern, mode='L')
        img.save(png_path)
        
        # Save as NPY for easy loading
        npy_path = os.path.join(output_dir, f"test_2x16_{name}.npy")
        np.save(npy_path, pattern)
        
        # Create expected output vector (one-hot for single sensors, multi-hot for combinations)
        expected = np.zeros(8, dtype=int)
        if 'high_body_temperature' in name:
            expected[0] = 1
        if 'muscle_fatigue' in name:
            expected[1] = 1
        if 'emg_fatigue' in name:
            expected[2] = 1
        if 'poor_posture' in name:
            expected[3] = 1
        if 'high_env_temperature' in name:
            expected[4] = 1
        if 'high_stress' in name:
            expected[5] = 1
        if 'eye_fatigue' in name:
            expected[6] = 1
        if name == 'normal':
            expected[7] = 1
        if 'all_alerts' in name:
            expected[0:7] = 1  # All alerts except normal
            
        test_info.append({
            'name': name,
            'png_file': png_path,
            'npy_file': npy_path,
            'expected_output': expected.tolist(),
            'pattern_shape': pattern.shape,
            'unique_values': len(np.unique(pattern))
        })
        
        print(f"  Created: {name} -> Expected: {expected}")
    
    # Save test manifest
    import json
    manifest_path = os.path.join(output_dir, "test_manifest_2x16.json")
    with open(manifest_path, 'w') as f:
        json.dump(test_info, f, indent=2)
    
    print(f"\\nTest images saved to: {output_dir}")
    print(f"Manifest saved to: {manifest_path}")
    
    return test_info

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate 2x16 test images for sensor classification")
    parser.add_argument("--output-dir", default="test_images_2x16", help="Output directory")
    args = parser.parse_args()
    
    test_info = generate_test_images_2x16(args.output_dir)
    print(f"\\nGenerated {len(test_info)} test cases in correct 2x16 format!")
