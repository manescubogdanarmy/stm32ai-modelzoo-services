"""
Script to generate synthetic images with colored shapes for CNN training.
"""
import os
import argparse
import random
import csv
from PIL import Image, ImageDraw

def random_color():
    """Generate a random RGB color."""
    return tuple(random.randint(0, 255) for _ in range(3))

def draw_shape(draw, shape, bbox, color):
    """Draw the specified shape into the given bounding box."""
    if shape == 'square':
        draw.rectangle(bbox, fill=color)
    elif shape == 'circle':
        draw.ellipse(bbox, fill=color)
    elif shape == 'triangle':
        left, top, right, bottom = bbox
        # Equilateral triangle
        points = [((left+right)/2, top), (left, bottom), (right, bottom)]
        draw.polygon(points, fill=color)
    else:
        raise ValueError(f"Unknown shape: {shape}")


def validate_image_generation(output_dir, sample_size=5):
    """Validate that generated images have correct sensor positions in 16x2 greyscale format."""
    print("\n=== VALIDATION REPORT ===")
    
    # Expected sensor positions in 16x2 image (each sensor gets 2x2 pixels)
    expected_positions = {
        'high_body_temperature': 0,  # Position 0: (0,0) to (1,1)
        'muscle_fatigue': 1,         # Position 1: (2,0) to (3,1)
        'emg_fatigue': 2,           # Position 2: (4,0) to (5,1)
        'poor_posture': 3,          # Position 3: (6,0) to (7,1)
        'high_env_temperature': 4,  # Position 4: (8,0) to (9,1)
        'high_stress': 5,           # Position 5: (10,0) to (11,1)
        'eye_fatigue': 6            # Position 6: (12,0) to (13,1)
    }
    
    for class_name, expected_pos in expected_positions.items():
        class_dir = os.path.join(output_dir, class_name)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) if f.endswith('.png')][:sample_size]
            
            for filename in files:
                filepath = os.path.join(class_dir, filename)
                img = Image.open(filepath)
                
                # Check if white pixels are at expected position (center of 2x2 square)
                center_x = expected_pos * 2 + 1  # Center of 2x2 square
                center_y = 1                     # Center row
                pixel_value = img.getpixel((center_x, center_y))
                
                # In greyscale mode, white = 255, black = 0
                is_white = pixel_value > 200
                status = "✓ CORRECT" if is_white else "✗ WRONG"
                print(f"{class_name}: {filename} - Position {expected_pos} {status}")
                
                if not is_white:
                    print(f"  Expected: White at position {expected_pos}, Found: {pixel_value}")
    
    # Check normal class - should have white at position 7 (normal state indicator)
    normal_dir = os.path.join(output_dir, 'normal')
    if os.path.exists(normal_dir):
        files = [f for f in os.listdir(normal_dir) if f.endswith('.png')][:sample_size]
        for filename in files:
            filepath = os.path.join(normal_dir, filename)
            img = Image.open(filepath)
            
            # Check that position 7 (normal indicator) is white
            normal_center_x = 7 * 2 + 1  # Position 7 center: (15,1)
            normal_center_y = 1
            pixel_value = img.getpixel((normal_center_x, normal_center_y))
            is_white = pixel_value > 200
            
            # Check that all sensor positions (0-6) are black
            all_sensors_black = True
            for pos in range(7):
                sensor_center_x = pos * 2 + 1
                sensor_center_y = 1
                sensor_pixel = img.getpixel((sensor_center_x, sensor_center_y))
                if sensor_pixel > 50:  # Should be black (close to 0)
                    all_sensors_black = False
                    break
            
            status = "✓ CORRECT" if (is_white and all_sensors_black) else "✗ WRONG"
            print(f"normal: {filename} - Normal indicator white and sensors black {status}")
    
    print("=== END VALIDATION ===\n")


def generate_images(output_dir, num_images, image_size):
    """Generate 16x2 greyscale images from simulated 7-sensor data and save CSV labels."""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'labels.csv')
    
    # CRITICAL: Define 7 sensors + 1 normal state in 16x2 grid (8 total positions)
    # Each sensor gets a 2x2 pixel square in the 16x2 image
    # Positions 0-7 map to (column*2, row) with 2x2 squares
    sensors = [
        ('temperature', 0, 'high_body_temperature'),     # Position 0 (0,0) to (1,1)
        ('myoware_muscle', 1, 'muscle_fatigue'),         # Position 1 (2,0) to (3,1)
        ('emg_electrodes', 2, 'emg_fatigue'),           # Position 2 (4,0) to (5,1)
        ('mpu9250_posture', 3, 'poor_posture'),         # Position 3 (6,0) to (7,1)
        ('sht31d_env_temp', 4, 'high_env_temperature'), # Position 4 (8,0) to (9,1)
        ('grove_gsr_stress', 5, 'high_stress'),         # Position 5 (10,0) to (11,1)
        ('eye_blink_fatigue', 6, 'eye_fatigue')         # Position 6 (12,0) to (13,1)
    ]
    
    # All possible classes (7 alerts + normal)
    classes = [sensor[2] for sensor in sensors] + ['normal']

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['filename', 'class'])
        
        # Generate equal distribution across all 8 classes
        class_counts = {cls: num_images // len(classes) for cls in classes}
        # Distribute remaining images randomly
        remaining = num_images % len(classes)
        for i in range(remaining):
            class_counts[classes[i]] += 1
        
        image_counter = 1
        
        for target_class in classes:
            for class_img_idx in range(class_counts[target_class]):
                # Create 16x2 greyscale image with black background (normal state)
                img = Image.new('L', (16, 2), 0)  # 'L' mode for greyscale, 0 = black
                draw = ImageDraw.Draw(img)
                
                # Fill all 7 sensor positions and normal position with BLACK (normal state)
                # Positions 0-6 for sensors, position 7 for normal state
                for idx in range(8):
                    x0 = idx * 2  # Each sensor gets 2 pixels width
                    y0 = 0         # All sensors are in the 2-row height
                    x1 = x0 + 2    # 2x2 square
                    y1 = 2         # Full height
                    bbox = (x0, y0, x1, y1)
                    draw.rectangle(bbox, fill=0)  # Black = normal
                
                # If alert class, set corresponding sensor position to WHITE
                if target_class != 'normal':
                    for sensor_name, position, class_name in sensors:
                        if class_name == target_class:
                            x0 = position * 2  # Each sensor gets 2 pixels width
                            y0 = 0             # All sensors are in the 2-row height
                            x1 = x0 + 2        # 2x2 square
                            y1 = 2             # Full height
                            bbox = (x0, y0, x1, y1)
                            draw.rectangle(bbox, fill=255)  # White = alert
                            break
                else:
                    # For normal class, set the normal position (position 7) to WHITE
                    x0 = 7 * 2  # Position 7 (14,0) to (15,1)
                    y0 = 0
                    x1 = x0 + 2
                    y1 = 2
                    bbox = (x0, y0, x1, y1)
                    draw.rectangle(bbox, fill=255)  # White = normal state active
                
                # Save in class-specific directory
                filename = f"sample_{image_counter:05d}.png"
                save_dir = os.path.join(output_dir, target_class)
                os.makedirs(save_dir, exist_ok=True)
                filepath = os.path.join(save_dir, filename)
                img.save(filepath)
                
                # Record in CSV
                writer.writerow([os.path.join(target_class, filename), target_class])
                image_counter += 1
    
    print(f"Generated {num_images} 16x2 greyscale images in '{output_dir}'.")
    print(f"Labels saved to {csv_path}.")
    print(f"Class distribution: {class_counts}")
    
    # Run validation
    validate_image_generation(output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic 16x2 greyscale sensor images.')
    parser.add_argument('--output_dir', type=str, default='generated',
                        help='Directory to save generated images and labels.')
    parser.add_argument('--num_images', type=int, default=100,
                        help='Number of images to generate.')
    parser.add_argument('--image_size', type=int, default=16,
                        help='Width of each image (height is fixed at 2).')
    args = parser.parse_args()
    generate_images(args.output_dir, args.num_images, args.image_size)
