"""
Generate a clean multi-label sensor dataset for proper training.
"""
import numpy as np
from PIL import Image
import os
import random
import json

def create_clean_sensor_dataset(output_dir="./datasets/sensor_data_multilabel", num_samples=1200):
    """
    Create a clean sensor dataset with proper multi-label patterns.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Sensor positions (each sensor occupies 2 pixels)
    sensor_positions = {
        0: (0, 1),    # high_body_temperature
        1: (2, 3),    # muscle_fatigue  
        2: (4, 5),    # emg_fatigue
        3: (6, 7),    # poor_posture
        4: (8, 9),    # high_env_temperature
        5: (10, 11),  # high_stress
        6: (12, 13),  # eye_fatigue
        # Position 14-15 reserved for normal indicator
    }
    
    class_names = ['high_body_temperature', 'muscle_fatigue', 'emg_fatigue', 'poor_posture', 
                   'high_env_temperature', 'high_stress', 'eye_fatigue', 'normal']
    
    samples = []
    
    print(f"Generating {num_samples} multi-label sensor samples...")
    
    for i in range(num_samples):
        # Create base pattern (all zeros)
        pattern = np.zeros((2, 16), dtype=np.uint8)
        
        # Randomly determine which sensors are active (30% chance per sensor)
        active_sensors = []
        for sensor_id in range(7):  # 0-6 are alert sensors
            if random.random() < 0.3:  # 30% chance each sensor is active
                active_sensors.append(sensor_id)
                start, end = sensor_positions[sensor_id]
                pattern[0, start:end+1] = 255
        
        # If no sensors are active, this is a "normal" state
        if len(active_sensors) == 0:
            pattern[0, 14:16] = 255  # Normal indicator
            active_sensors = [7]  # Normal class
        
        # Create multi-label vector
        label_vector = np.zeros(8, dtype=int)
        for sensor_id in active_sensors:
            label_vector[sensor_id] = 1
        
        # Save the pattern
        filename = f"sample_{i:05d}"
        
        # Save as PNG
        img = Image.fromarray(pattern, mode='L')
        img_path = os.path.join(output_dir, f"{filename}.png")
        img.save(img_path)
        
        # Save as NPY for easy loading
        npy_path = os.path.join(output_dir, f"{filename}.npy")
        np.save(npy_path, pattern)
        
        # Store sample info
        samples.append({
            'filename': filename,
            'image_path': img_path,
            'npy_path': npy_path,
            'active_sensors': active_sensors,
            'label_vector': label_vector.tolist(),
            'class_names': [class_names[i] for i in active_sensors]
        })
        
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples")
    
    # Save dataset manifest
    manifest = {
        'num_samples': len(samples),
        'num_classes': 8,
        'class_names': class_names,
        'sensor_positions': sensor_positions,
        'samples': samples,
        'statistics': {
            'single_label_samples': sum(1 for s in samples if sum(s['label_vector']) == 1),
            'multi_label_samples': sum(1 for s in samples if sum(s['label_vector']) > 1),
            'normal_samples': sum(1 for s in samples if s['label_vector'][7] == 1),
            'class_distribution': [sum(1 for s in samples if s['label_vector'][i] == 1) for i in range(8)]
        }
    }
    
    manifest_path = os.path.join(output_dir, "dataset_manifest.json")
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\\nDataset created successfully!")
    print(f"  Total samples: {manifest['num_samples']}")
    print(f"  Single-label samples: {manifest['statistics']['single_label_samples']}")
    print(f"  Multi-label samples: {manifest['statistics']['multi_label_samples']}")
    print(f"  Normal samples: {manifest['statistics']['normal_samples']}")
    print(f"  Class distribution: {manifest['statistics']['class_distribution']}")
    print(f"  Manifest saved to: {manifest_path}")
    
    return manifest

if __name__ == "__main__":
    create_clean_sensor_dataset()
