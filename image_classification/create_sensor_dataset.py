"""
Script to create a sensor dataset for image classification training.
Generates 16x2 greyscale images organized in class directories as required by STM32AI model zoo.
"""
import os
import argparse
import random
import csv
from PIL import Image, ImageDraw

def create_sensor_dataset(output_dir, num_images_per_class):
    """Create a dataset with 8 classes in proper directory structure for image classification."""
    
    # Define the 8 classes (7 sensors + normal state)
    classes = [
        'high_body_temperature',
        'muscle_fatigue', 
        'emg_fatigue',
        'poor_posture',
        'high_env_temperature',
        'high_stress',
        'eye_fatigue',
        'normal'
    ]
    
    # Create main dataset directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create CSV for tracking dataset info
    csv_path = os.path.join(output_dir, 'dataset_info.csv')
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['class', 'image_count', 'sensor_position'])
        
        # Generate images for each class
        for class_idx, class_name in enumerate(classes):
            # Create class directory
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            print(f"Generating {num_images_per_class} images for class '{class_name}'...")
            
            # Generate images for this class
            for img_idx in range(num_images_per_class):
                # Create 16x2 greyscale image with black background
                img = Image.new('L', (16, 2), 0)  # 'L' mode for greyscale, 0 = black
                draw = ImageDraw.Draw(img)
                
                if class_name == 'normal':
                    # For normal class, activate position 7 (normal indicator)
                    x0 = 7 * 2  # Position 7: (14,0) to (15,1)
                    y0 = 0
                    x1 = x0 + 2
                    y1 = 2
                    bbox = (x0, y0, x1, y1)
                    draw.rectangle(bbox, fill=255)  # White = normal state active
                    sensor_position = 7
                else:
                    # For alert classes, activate the corresponding sensor position
                    sensor_position = class_idx  # Position 0-6 for sensors
                    x0 = sensor_position * 2  # Each sensor gets 2 pixels width
                    y0 = 0                     # All sensors are in the 2-row height
                    x1 = x0 + 2                # 2x2 square
                    y1 = 2                     # Full height
                    bbox = (x0, y0, x1, y1)
                    draw.rectangle(bbox, fill=255)  # White = alert
                
                # Add slight variation to make each image unique
                # Add small random noise to some black pixels to create variety
                for noise_x in range(16):
                    for noise_y in range(2):
                        # Skip the main signal area
                        signal_area = False
                        if class_name == 'normal' and 14 <= noise_x <= 15:
                            signal_area = True
                        elif class_name != 'normal' and sensor_position*2 <= noise_x <= sensor_position*2+1:
                            signal_area = True
                        
                        if not signal_area and random.random() < 0.05:  # 5% chance for noise
                            current_pixel = img.getpixel((noise_x, noise_y))
                            if current_pixel == 0:  # Only add noise to black pixels
                                noise_value = random.randint(1, 15)  # Very low noise
                                img.putpixel((noise_x, noise_y), noise_value)
                
                # Save image
                filename = f"{class_name}_{img_idx+1:05d}.png"
                filepath = os.path.join(class_dir, filename)
                img.save(filepath)
            
            # Record class info in CSV
            writer.writerow([class_name, num_images_per_class, 
                           sensor_position if class_name != 'normal' else 7])
            
            print(f"  ✓ Created {num_images_per_class} images in {class_dir}")
    
    print(f"\n✓ Dataset created successfully in '{output_dir}'")
    print(f"✓ Classes: {len(classes)}")
    print(f"✓ Images per class: {num_images_per_class}")
    print(f"✓ Total images: {len(classes) * num_images_per_class}")
    print(f"✓ Dataset info saved to: {csv_path}")
    
    # Validate the dataset
    validate_dataset(output_dir, classes)
    
    return output_dir, classes

def validate_dataset(dataset_dir, expected_classes):
    """Validate that the dataset structure is correct."""
    print("\n=== DATASET VALIDATION ===")
    
    all_good = True
    
    for class_name in expected_classes:
        class_dir = os.path.join(dataset_dir, class_name)
        
        if not os.path.exists(class_dir):
            print(f"✗ Missing class directory: {class_name}")
            all_good = False
            continue
        
        # Count images in class directory
        image_files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
        print(f"✓ {class_name}: {len(image_files)} images")
        
        # Validate a sample image
        if image_files:
            sample_file = os.path.join(class_dir, image_files[0])
            img = Image.open(sample_file)
            
            if img.size != (16, 2):
                print(f"  ✗ Wrong image size: {img.size}, expected (16, 2)")
                all_good = False
            elif img.mode != 'L':
                print(f"  ✗ Wrong image mode: {img.mode}, expected 'L' (greyscale)")
                all_good = False
            else:
                print(f"  ✓ Sample image format correct")
    
    if all_good:
        print("✓ Dataset validation passed!")
    else:
        print("✗ Dataset validation failed!")
    
    print("=== END VALIDATION ===\n")

def create_training_config(dataset_dir, classes, num_epochs=50):
    """Create a training-only configuration file."""
    
    config_content = f"""general:
  project_name: sensor_classification
  logs_dir: logs
  saved_models_dir: saved_models
  display_figures: True
  global_seed: 123
  gpu_memory_limit: 8

operation_mode: training

dataset:
  name: sensor_data
  class_names: {classes}
  training_path: {dataset_dir}
  validation_path:
  validation_split: 0.2
  test_path:
  check_image_files: True
  seed: 123

preprocessing:
  rescaling:
    scale: 1/255.0
    offset: 0
  resizing:
    aspect_ratio: fit
    interpolation: nearest
  color_mode: grayscale

data_augmentation:
  # Minimal augmentation for sensor data to preserve signal integrity
  random_flip:
    mode: horizontal
  random_translation:
    width_factor: 0.1
    height_factor: 0.0  # No vertical translation for 2-pixel height
    fill_mode: constant
    interpolation: nearest

training:
  model:
    name: mobilenet
    version: v2
    alpha: 0.35
    input_shape: (16, 2, 1)  # 16x2 greyscale
    pretrained_weights: None  # No pretrained weights for custom sensor data
  batch_size: 32
  epochs: {num_epochs}
  dropout: 0.3
  optimizer:
    Adam:
      learning_rate: 0.001
  callbacks:
    ReduceLROnPlateau:
      monitor: val_accuracy
      mode: max
      factor: 0.5
      patience: 10
      min_lr: 1.0e-05
    EarlyStopping:
      monitor: val_accuracy
      mode: max
      restore_best_weights: true
      patience: 20

mlflow:
  uri: ./src/experiments_outputs/mlruns

hydra:
  run:
    dir: ./src/experiments_outputs/${{now:%Y_%m_%d_%H_%M_%S}}
"""
    
    config_path = os.path.join(os.path.dirname(dataset_dir), 'sensor_training_config.yaml')
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Training configuration saved to: {config_path}")
    return config_path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create sensor dataset for STM32AI image classification.')
    parser.add_argument('--output_dir', type=str, default='./datasets/sensor_data',
                        help='Directory to save the dataset.')
    parser.add_argument('--num_images_per_class', type=int, default=500,
                        help='Number of images to generate per class.')
    parser.add_argument('--create_config', action='store_true',
                        help='Create training configuration file.')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs for config file.')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SENSOR DATASET CREATOR FOR STM32AI MODEL ZOO")
    print("="*60)
    
    # Create the dataset
    dataset_dir, classes = create_sensor_dataset(args.output_dir, args.num_images_per_class)
    
    # Create training config if requested
    if args.create_config:
        config_path = create_training_config(dataset_dir, classes, args.epochs)
        print(f"\nTo start training, run:")
        print(f"python stm32ai_main.py --config-path=\"{os.path.dirname(config_path)}\" --config-name=\"{os.path.basename(config_path)}\"")
