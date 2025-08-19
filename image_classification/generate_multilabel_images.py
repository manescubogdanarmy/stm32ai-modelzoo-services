"""
Script to generate multi-label images with multiple active sensors for CNN training.
This complements the single-label generator by creating realistic multi-sensor scenarios.
"""
import os
import argparse
import random
import csv
import itertools
from PIL import Image, ImageDraw

def generate_multilabel_images(output_dir, num_images, image_size):
    """Generate 16x2 greyscale images with multiple active sensors and save multi-label CSV."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'multilabel_data.csv')
    
    # CRITICAL: Define 7 sensors with EXACT position mapping for 16x2 grid
    # Each sensor gets a 2x2 pixel area in the 16x2 image
    # Position 7 is reserved for normal state indicator
    sensors = [
        ('high_body_temperature', 0),   # Position 0: (0,0) to (1,1)
        ('muscle_fatigue', 1),          # Position 1: (2,0) to (3,1)
        ('emg_fatigue', 2),            # Position 2: (4,0) to (5,1)
        ('poor_posture', 3),           # Position 3: (6,0) to (7,1)
        ('high_env_temperature', 4),   # Position 4: (8,0) to (9,1)
        ('high_stress', 5),            # Position 5: (10,0) to (11,1)
        ('eye_fatigue', 6)             # Position 6: (12,0) to (13,1)
    ]
    
    # Generate all possible combinations (1 to 7 active sensors)
    all_combinations = []
    for r in range(1, len(sensors) + 1):
        for combo in itertools.combinations(sensors, r):
            all_combinations.append(combo)
    
    # Add normal case (no active sensors)
    all_combinations.append(())
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # CSV header: filename + binary columns for each sensor + normal indicator
        header = ['filename'] + [name for name, _ in sensors] + ['normal']
        writer.writerow(header)
        
        for i in range(1, num_images + 1):
            # Choose random combination (weighted toward fewer active sensors for realism)
            weights = [4**(-len(combo)) if combo else 0.15 for combo in all_combinations]
            active_combo = random.choices(all_combinations, weights=weights)[0]
            
            # Create 16x2 greyscale image with black background
            img = Image.new('L', (16, 2), 0)  # 'L' mode for greyscale, 0 = black
            draw = ImageDraw.Draw(img)
            
            # Set active sensors to WHITE (alert state)
            active_sensor_names = []
            for sensor_name, position in active_combo:
                active_sensor_names.append(sensor_name)
                x0 = position * 2  # Each sensor gets 2 pixels width
                y0 = 0             # All sensors are in the 2-row height
                x1 = x0 + 2        # 2x2 square
                y1 = 2             # Full height
                bbox = (x0, y0, x1, y1)
                draw.rectangle(bbox, fill=255)  # White = alert
            
            # Create binary label vector for 7 sensors
            sensor_labels = []
            for sensor_name, _ in sensors:
                sensor_labels.append(1 if sensor_name in active_sensor_names else 0)
            
            # Add normal state indicator (1 if no sensors active, 0 otherwise)
            normal_label = 1 if not active_combo else 0
            if normal_label:
                # Draw normal indicator at position 7
                x0 = 7 * 2  # Position 7: (14,0) to (15,1)
                y0 = 0
                x1 = x0 + 2
                y1 = 2
                bbox = (x0, y0, x1, y1)
                draw.rectangle(bbox, fill=255)  # White = normal state active
            
            # Combine all labels
            all_labels = sensor_labels + [normal_label]
            
            # Generate descriptive filename
            if not active_combo:
                subdir = 'normal'
                filename = f"normal_{i:05d}.png"
            else:
                combo_str = "_".join([name.replace('_', '') for name, _ in active_combo])
                subdir = f"combo_{len(active_combo)}"
                filename = f"{combo_str}_{i:05d}.png"
            
            # Save image
            save_dir = os.path.join(output_dir, subdir)
            os.makedirs(save_dir, exist_ok=True)
            filepath = os.path.join(save_dir, filename)
            img.save(filepath)
            
            # Write CSV row: filename + binary labels
            writer.writerow([os.path.join(subdir, filename)] + all_labels)
    
    print(f"Generated {num_images} 16x2 greyscale multi-label images in '{output_dir}'.")
    print(f"Multi-label data saved to {csv_path}.")
    print(f"Combinations range from 0 (normal) to {len(sensors)} active sensors.")
    
    # Run validation
    validate_multilabel_generation(output_dir)

def generate_specific_combinations(output_dir, image_size):
    """Generate specific realistic multi-sensor combinations in 16x2 greyscale format."""
    realistic_combos = [
        # Common co-occurring conditions
        (['high_stress', 'eye_fatigue'], 'stress_eye', 50),
        (['muscle_fatigue', 'poor_posture'], 'muscle_posture', 40),
        (['high_body_temperature', 'high_env_temperature'], 'heat_related', 30),
        (['emg_fatigue', 'muscle_fatigue', 'poor_posture'], 'physical_strain', 25),
        (['high_stress', 'eye_fatigue', 'muscle_fatigue'], 'work_overload', 20),
        (['high_body_temperature', 'high_stress', 'muscle_fatigue'], 'exercise_stress', 15),
        # Extreme case - all sensors active
        (['high_body_temperature', 'muscle_fatigue', 'emg_fatigue', 'poor_posture', 
          'high_env_temperature', 'high_stress', 'eye_fatigue'], 'critical_state', 5)
    ]
    
    # CRITICAL: Correct sensor position mapping for 16x2 grid
    sensors_map = {
        'high_body_temperature': 0,  # Position 0: (0,0) to (1,1)
        'muscle_fatigue': 1,         # Position 1: (2,0) to (3,1)
        'emg_fatigue': 2,           # Position 2: (4,0) to (5,1)
        'poor_posture': 3,          # Position 3: (6,0) to (7,1)
        'high_env_temperature': 4,  # Position 4: (8,0) to (9,1)
        'high_stress': 5,           # Position 5: (10,0) to (11,1)
        'eye_fatigue': 6            # Position 6: (12,0) to (13,1)
    }
    
    csv_path = os.path.join(output_dir, 'realistic_combinations.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        header = ['filename', 'scenario'] + list(sensors_map.keys()) + ['normal']
        writer.writerow(header)
        
        for active_sensors, scenario, count in realistic_combos:
            subdir = f"realistic_{scenario}"
            save_dir = os.path.join(output_dir, subdir)
            os.makedirs(save_dir, exist_ok=True)
            
            for i in range(count):
                # Create 16x2 greyscale image with black background
                img = Image.new('L', (16, 2), 0)  # 'L' mode for greyscale, 0 = black
                draw = ImageDraw.Draw(img)
                
                # Set active sensors to WHITE (alert state)
                sensor_labels = [0] * 7
                for sensor_name in active_sensors:
                    position = sensors_map[sensor_name]
                    sensor_labels[position] = 1
                    x0 = position * 2  # Each sensor gets 2 pixels width
                    y0 = 0             # All sensors are in the 2-row height
                    x1 = x0 + 2        # 2x2 square
                    y1 = 2             # Full height
                    bbox = (x0, y0, x1, y1)
                    draw.rectangle(bbox, fill=255)  # White = alert
                
                # Normal state indicator (0 since we have active sensors)
                normal_label = 0
                all_labels = sensor_labels + [normal_label]
                
                filename = f"{scenario}_{i+1:03d}.png"
                filepath = os.path.join(save_dir, filename)
                img.save(filepath)
                
                writer.writerow([os.path.join(subdir, filename), scenario] + all_labels)
    
    print(f"Generated realistic combinations in '{output_dir}' with scenario-based folders.")
    print(f"Realistic combinations data saved to {csv_path}.")


def validate_multilabel_generation(output_dir, sample_size=3):
    """Validate that generated multi-label 16x2 greyscale images have correct sensor positions."""
    print("\n=== MULTI-LABEL VALIDATION REPORT ===")
    
    csv_path = os.path.join(output_dir, 'multilabel_data.csv')
    if not os.path.exists(csv_path):
        print("No multilabel_data.csv found for validation.")
        return
    
    import csv
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        sensor_names = [col for col in reader.fieldnames if col not in ['filename', 'normal']]
        
        # Sample a few rows for validation
        rows = list(reader)
        sample_rows = rows[:sample_size] if len(rows) >= sample_size else rows
        
        for row in sample_rows:
            filepath = os.path.join(output_dir, row['filename'])
            if not os.path.exists(filepath):
                print(f"File not found: {filepath}")
                continue
                
            img = Image.open(filepath)
            
            print(f"\nValidating: {row['filename']}")
            
            all_correct = True
            for i, sensor_name in enumerate(sensor_names):
                expected_active = int(row[sensor_name])
                
                # Check actual pixel at sensor position center
                center_x = i * 2 + 1  # Center of 2x2 square
                center_y = 1          # Center row
                pixel_value = img.getpixel((center_x, center_y))
                
                is_white = pixel_value > 200
                is_black = pixel_value < 50
                
                if expected_active == 1 and not is_white:
                    print(f"  ✗ {sensor_name}: Expected WHITE, found {pixel_value}")
                    all_correct = False
                elif expected_active == 0 and not is_black:
                    print(f"  ✗ {sensor_name}: Expected BLACK, found {pixel_value}")
                    all_correct = False
                else:
                    status = "WHITE" if expected_active == 1 else "BLACK"
                    print(f"  ✓ {sensor_name}: Correct {status}")
            
            # Check normal indicator
            normal_expected = int(row['normal'])
            normal_center_x = 7 * 2 + 1  # Position 7 center
            normal_center_y = 1
            normal_pixel = img.getpixel((normal_center_x, normal_center_y))
            
            normal_is_white = normal_pixel > 200
            normal_is_black = normal_pixel < 50
            
            if normal_expected == 1 and not normal_is_white:
                print(f"  ✗ normal: Expected WHITE, found {normal_pixel}")
                all_correct = False
            elif normal_expected == 0 and not normal_is_black:
                print(f"  ✗ normal: Expected BLACK, found {normal_pixel}")
                all_correct = False
            else:
                status = "WHITE" if normal_expected == 1 else "BLACK"
                print(f"  ✓ normal: Correct {status}")
            
            if all_correct:
                print(f"  ✓ ALL POSITIONS CORRECT for {row['filename']}")
    
    print("=== END MULTI-LABEL VALIDATION ===\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate multi-label 16x2 greyscale sensor images.')
    parser.add_argument('--output_dir', type=str, default='generated_multilabel',
                        help='Directory to save generated images and labels.')
    parser.add_argument('--num_images', type=int, default=1000,
                        help='Number of random combination images to generate.')
    parser.add_argument('--image_size', type=int, default=16,
                        help='Width of each image (height is fixed at 2).')
    parser.add_argument('--mode', type=str, choices=['random', 'realistic', 'both'], default='both',
                        help='Generation mode: random combinations, realistic scenarios, or both.')
    
    args = parser.parse_args()
    
    if args.mode in ['random', 'both']:
        generate_multilabel_images(args.output_dir, args.num_images, args.image_size)
    
    if args.mode in ['realistic', 'both']:
        generate_specific_combinations(args.output_dir, args.image_size)
        # Run validation for realistic combinations too
        validate_multilabel_generation(args.output_dir)
