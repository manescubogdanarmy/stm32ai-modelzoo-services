"""
Test image generator for 32x32 pixel images compatible with the trained sensor classification model.
Creates test images in the same format the model expects for inference.
"""
import os
import argparse
import numpy as np
from PIL import Image, ImageDraw
import sys

def display_sensor_grid():
    """Display the sensor layout in 32x32 format."""
    print("\n" + "="*60)
    print("SENSOR TEST IMAGE GENERATOR (32x32 Format)")
    print("="*60)
    print("This generates 32x32 greyscale images for testing the trained model.")
    print("Each sensor occupies a 4x32 pixel region (vertical strips):")
    print("")
    print("Sensor Position:  Region:      Sensor Name:")
    print("Position 0       (0-3,0-31)   high_body_temperature")
    print("Position 1       (4-7,0-31)   muscle_fatigue") 
    print("Position 2       (8-11,0-31)  emg_fatigue")
    print("Position 3       (12-15,0-31) poor_posture")
    print("Position 4       (16-19,0-31) high_env_temperature")
    print("Position 5       (20-23,0-31) high_stress")
    print("Position 6       (24-27,0-31) eye_fatigue")
    print("Position 7       (28-31,0-31) normal_state")
    print("="*60)
    print("Visual Layout (32x32 grid, each | = 4 pixels wide):")
    print("┌────┬────┬────┬────┬────┬────┬────┬────┐")
    print("│ 0  │ 1  │ 2  │ 3  │ 4  │ 5  │ 6  │ 7  │")
    print("│    │    │    │    │    │    │    │    │")
    print("│...32 rows high...                    │")
    print("└────┴────┴────┴────┴────┴────┴────┴────┘")
    print("Colors: BLACK (0) = Normal/Inactive, WHITE (255) = Alert/Active")
    print("="*60)

def get_user_input():
    """Get user input for sensor states."""
    sensors = [
        ('high_body_temperature', 0),
        ('muscle_fatigue', 1),
        ('emg_fatigue', 2),
        ('poor_posture', 3),
        ('high_env_temperature', 4),
        ('high_stress', 5),
        ('eye_fatigue', 6)
    ]
    
    print("\nSELECT SENSOR STATES:")
    print("Enter 'w' for WHITE (alert), 'b' for BLACK (normal), or 'q' to quit")
    print("-" * 50)
    
    sensor_states = {}
    
    for sensor_name, position in sensors:
        while True:
            user_input = input(f"Sensor {position} ({sensor_name}): ").lower().strip()
            
            if user_input == 'q':
                print("Exiting...")
                return None
            elif user_input in ['w', 'white']:
                sensor_states[position] = 'white'
                print(f"  ✓ Position {position} set to WHITE (alert)")
                break
            elif user_input in ['b', 'black']:
                sensor_states[position] = 'black'
                print(f"  ✓ Position {position} set to BLACK (normal)")
                break
            else:
                print("  Invalid input! Enter 'w' for white, 'b' for black, or 'q' to quit")
    
    return sensor_states

def get_batch_input():
    """Get batch input for multiple sensor states."""
    print("\nBATCH INPUT MODE:")
    print("Enter sensor states as comma-separated values (0-6)")
    print("Format: 'w,b,w,b,w,b,w' for positions 0,1,2,3,4,5,6")
    print("Or enter positions to be WHITE: '0,2,5' (others will be black)")
    print("Or 'all' for all white, 'none' for all black")
    print("-" * 50)
    
    while True:
        user_input = input("Enter sensor states: ").strip()
        
        if user_input.lower() == 'q':
            return None
        elif user_input.lower() == 'all':
            return {i: 'white' for i in range(7)}
        elif user_input.lower() == 'none':
            return {i: 'black' for i in range(7)}
        
        # Try parsing as positions to be white
        if ',' in user_input and all(c in '0123456, ' for c in user_input):
            try:
                white_positions = [int(x.strip()) for x in user_input.split(',') if x.strip()]
                if all(0 <= pos <= 6 for pos in white_positions):
                    sensor_states = {i: 'black' for i in range(7)}
                    for pos in white_positions:
                        sensor_states[pos] = 'white'
                    return sensor_states
                else:
                    print("  Invalid positions! Use numbers 0-6 only.")
                    continue
            except ValueError:
                pass
        
        # Try parsing as full state list
        if ',' in user_input:
            states = [x.strip().lower() for x in user_input.split(',')]
            if len(states) == 7 and all(s in ['w', 'b', 'white', 'black'] for s in states):
                sensor_states = {}
                for i, state in enumerate(states):
                    sensor_states[i] = 'white' if state in ['w', 'white'] else 'black'
                return sensor_states
        
        print("  Invalid format! Try again or enter 'q' to quit.")

def create_test_image_32x32(sensor_states):
    """Create 32x32 greyscale test image based on sensor states."""
    # Create 32x32 greyscale image with black background
    img = Image.new('L', (32, 32), 0)  # 'L' mode for greyscale, 0 = black
    draw = ImageDraw.Draw(img)
    
    # Each sensor gets a 4-pixel wide vertical strip (4x32 pixels)
    for position in range(7):
        x0 = position * 4  # Each sensor gets 4 pixels width
        y0 = 0             # Full height from top
        x1 = x0 + 4        # 4 pixels wide
        y1 = 32            # Full 32 pixel height
        bbox = (x0, y0, x1, y1)
        
        if sensor_states.get(position, 'black') == 'white':
            draw.rectangle(bbox, fill=255)  # White = alert
        else:
            draw.rectangle(bbox, fill=0)    # Black = normal
    
    # Handle normal state indicator at position 7 (rightmost strip)
    all_sensors_normal = all(sensor_states.get(i, 'black') == 'black' for i in range(7))
    x0 = 7 * 4  # Position 7: pixels 28-31
    y0 = 0
    x1 = x0 + 4  # 4 pixels wide (28-31)
    y1 = 32      # Full height
    bbox = (x0, y0, x1, y1)
    
    if all_sensors_normal:
        # If all sensors are normal (black), set normal indicator to white
        draw.rectangle(bbox, fill=255)  # White = normal state active
    else:
        # If any sensor is alert, normal indicator is black
        draw.rectangle(bbox, fill=0)    # Black = normal state inactive
    
    return img

def create_test_image_numpy(sensor_states):
    """Create 32x32 numpy array for direct model inference."""
    # Create 32x32 array with values 0-255 (will be normalized by model)
    img_array = np.zeros((32, 32), dtype=np.uint8)
    
    # Fill sensor positions
    for position in range(7):
        x_start = position * 4
        x_end = x_start + 4
        
        if sensor_states.get(position, 'black') == 'white':
            img_array[:, x_start:x_end] = 255  # White = alert
        else:
            img_array[:, x_start:x_end] = 0    # Black = normal
    
    # Handle normal state indicator
    all_sensors_normal = all(sensor_states.get(i, 'black') == 'black' for i in range(7))
    x_start = 7 * 4  # Position 7: columns 28-31
    x_end = x_start + 4
    
    if all_sensors_normal:
        img_array[:, x_start:x_end] = 255  # White = normal state active
    else:
        img_array[:, x_start:x_end] = 0    # Black = normal state inactive
    
    return img_array

def display_summary(sensor_states):
    """Display summary of selected sensor states."""
    sensor_names = [
        'high_body_temperature', 'muscle_fatigue', 'emg_fatigue',
        'poor_posture', 'high_env_temperature', 'high_stress', 'eye_fatigue'
    ]
    
    print("\n" + "="*50)
    print("SELECTED SENSOR STATES:")
    print("="*50)
    
    white_sensors = []
    black_sensors = []
    
    for position in range(7):
        state = sensor_states.get(position, 'black')
        sensor_name = sensor_names[position]
        status = "⚪ WHITE (ALERT)" if state == 'white' else "⚫ BLACK (NORMAL)"
        print(f"Position {position}: {sensor_name:20} -> {status}")
        
        if state == 'white':
            white_sensors.append(f"{position}({sensor_name})")
        else:
            black_sensors.append(f"{position}({sensor_name})")
    
    print("-" * 50)
    print(f"ACTIVE ALERTS: {len(white_sensors)} sensors")
    if white_sensors:
        print(f"White sensors: {', '.join(white_sensors)}")
    print(f"NORMAL STATE: {len(black_sensors)} sensors") 
    if black_sensors:
        print(f"Black sensors: {', '.join(black_sensors)}")
    
    # Check if normal state indicator should be active
    all_normal = len(white_sensors) == 0
    print("-" * 50)
    print(f"NORMAL INDICATOR (Position 7): {'⚪ WHITE (ACTIVE)' if all_normal else '⚫ BLACK (INACTIVE)'}")
    print("="*50)

def generate_multilabel_vector(sensor_states):
    """Generate multi-label binary vector for model prediction comparison."""
    # This should match the training data format
    vector = []
    
    # First 7 positions for sensors (binary: 1=alert, 0=normal)
    for position in range(7):
        vector.append(1 if sensor_states.get(position, 'black') == 'white' else 0)
    
    # 8th position for normal state (1 if all sensors are normal, 0 otherwise)
    all_normal = all(sensor_states.get(i, 'black') == 'black' for i in range(7))
    vector.append(1 if all_normal else 0)
    
    return vector

def save_test_image(img, sensor_states, output_dir, save_numpy=False):
    """Save the test image with descriptive filename."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename based on active sensors
    white_positions = [str(pos) for pos in range(7) if sensor_states.get(pos, 'black') == 'white']
    
    if not white_positions:
        base_filename = "test_all_normal"
    elif len(white_positions) == 7:
        base_filename = "test_all_alert"
    else:
        base_filename = f"test_sensors_{'_'.join(white_positions)}"
    
    # Save PNG image
    png_filepath = os.path.join(output_dir, f"{base_filename}.png")
    img.save(png_filepath)
    
    filepaths = [png_filepath]
    
    # Optionally save as numpy array for direct model loading
    if save_numpy:
        numpy_array = create_test_image_numpy(sensor_states)
        npy_filepath = os.path.join(output_dir, f"{base_filename}.npy")
        np.save(npy_filepath, numpy_array)
        filepaths.append(npy_filepath)
    
    return filepaths

def interactive_mode(output_dir):
    """Run interactive mode for single image generation."""
    display_sensor_grid()
    
    while True:
        print("\n" + "="*60)
        print("INTERACTIVE TEST IMAGE GENERATOR (32x32)")
        print("="*60)
        print("Options:")
        print("1. Individual sensor selection")
        print("2. Batch input mode")
        print("3. Show sensor grid layout")
        print("4. Quit")
        print("-" * 60)
        
        choice = input("Select option (1-4): ").strip()
        
        if choice == '1':
            sensor_states = get_user_input()
        elif choice == '2':
            sensor_states = get_batch_input()
        elif choice == '3':
            display_sensor_grid()
            continue
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice! Please enter 1-4.")
            continue
        
        if sensor_states is None:
            continue
        
        # Display summary
        display_summary(sensor_states)
        
        # Generate 32x32 image
        img = create_test_image_32x32(sensor_states)
        
        # Save image (with optional numpy array)
        save_numpy = input("\nSave numpy array for direct model loading? (y/n): ").lower().strip() in ['y', 'yes']
        filepaths = save_test_image(img, sensor_states, output_dir, save_numpy)
        
        # Generate expected output vector
        ml_vector = generate_multilabel_vector(sensor_states)
        
        print(f"\n✓ Image saved: {filepaths[0]}")
        if len(filepaths) > 1:
            print(f"✓ Numpy array saved: {filepaths[1]}")
        print(f"✓ Image size: 32x32 pixels (greyscale)")
        print(f"✓ Expected model output: {ml_vector}")
        print(f"✓ Binary representation: {''.join(map(str, ml_vector))}")
        
        # Show model inference hint
        print("\n" + "="*50)
        print("MODEL INFERENCE HINTS:")
        print("="*50)
        print("To test with your trained model:")
        print("1. Load the image and resize to (1, 32, 32, 1) for batch inference")
        print("2. Normalize pixel values to [0,1] range (divide by 255)")
        print("3. Compare model output with expected vector above")
        print(f"4. Model path: {output_dir}/../src/experiments_outputs/2025_08_01_12_59_58/saved_models/best_augmented_model.h5")
        print("="*50)
        
        # Ask if user wants to generate another
        continue_choice = input("\nGenerate another image? (y/n): ").lower().strip()
        if continue_choice not in ['y', 'yes']:
            break

def batch_mode(output_dir):
    """Generate a batch of predefined test cases for comprehensive model testing."""
    print("\n" + "="*60)
    print("BATCH MODE: Generating comprehensive test cases (32x32)")
    print("="*60)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    test_cases = [
        # Single sensor tests
        ({0: 'white'}, "single_body_temp"),
        ({1: 'white'}, "single_muscle_fatigue"),
        ({2: 'white'}, "single_emg_fatigue"),
        ({3: 'white'}, "single_poor_posture"),
        ({4: 'white'}, "single_env_temp"),
        ({5: 'white'}, "single_stress"),
        ({6: 'white'}, "single_eye_fatigue"),
        
        # Two sensor combinations
        ({0: 'white', 4: 'white'}, "heat_related"),
        ({1: 'white', 3: 'white'}, "muscle_posture"),
        ({5: 'white', 6: 'white'}, "stress_eye"),
        ({1: 'white', 2: 'white'}, "muscle_emg"),
        ({0: 'white', 5: 'white'}, "temp_stress"),
        
        # Three sensor combinations
        ({1: 'white', 2: 'white', 3: 'white'}, "physical_strain"),
        ({5: 'white', 6: 'white', 1: 'white'}, "work_overload"),
        ({0: 'white', 4: 'white', 5: 'white'}, "environmental_stress"),
        
        # More complex combinations
        ({0: 'white', 1: 'white', 5: 'white', 6: 'white'}, "high_stress_state"),
        ({1: 'white', 2: 'white', 3: 'white', 6: 'white'}, "work_fatigue"),
        ({0: 'white', 3: 'white', 4: 'white', 5: 'white'}, "environmental_physical"),
        
        # Edge cases
        ({i: 'black' for i in range(7)}, "all_normal"),
        ({i: 'white' for i in range(7)}, "all_alert"),
    ]
    
    # Add default black state for missing positions
    for states, name in test_cases:
        for i in range(7):
            if i not in states:
                states[i] = 'black'
    
    generated_files = []
    
    for sensor_states, test_name in test_cases:
        print(f"\nGenerating: {test_name}")
        
        # Create 32x32 image
        img = create_test_image_32x32(sensor_states)
        
        # Save both PNG and numpy array
        filepaths = save_test_image(img, sensor_states, output_dir, save_numpy=True)
        
        # Generate expected output
        ml_vector = generate_multilabel_vector(sensor_states)
        white_count = sum(ml_vector[:-1])  # Exclude normal indicator from count
        
        print(f"  ✓ Saved PNG: {os.path.basename(filepaths[0])}")
        print(f"  ✓ Saved NPY: {os.path.basename(filepaths[1])}")
        print(f"  ✓ Active sensors: {white_count}/7")
        print(f"  ✓ Expected output: {ml_vector}")
        
        generated_files.append((filepaths, ml_vector, test_name, sensor_states))
    
    # Save comprehensive test manifest
    manifest_path = os.path.join(output_dir, "test_manifest_32x32.csv")
    with open(manifest_path, 'w') as f:
        f.write("png_filename,npy_filename,test_name,expected_output,active_sensors,normal_indicator,sensor_states\n")
        for filepaths, ml_vector, test_name, sensor_states in generated_files:
            png_name = os.path.basename(filepaths[0])
            npy_name = os.path.basename(filepaths[1])
            vector_str = '[' + ','.join(map(str, ml_vector)) + ']'
            active_sensors = sum(ml_vector[:-1])
            normal_indicator = ml_vector[-1]
            states_str = '{' + ','.join(f'{k}:{v}' for k, v in sensor_states.items()) + '}'
            f.write(f"{png_name},{npy_name},{test_name},\"{vector_str}\",{active_sensors},{normal_indicator},\"{states_str}\"\n")
    
    # Save inference script template
    inference_script_path = os.path.join(output_dir, "test_model_inference.py")
    with open(inference_script_path, 'w') as f:
        f.write('''"""
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
    
    # Normalize to [0, 1] range
    img_array = img_array.astype(np.float32) / 255.0
    
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
''')
    
    print(f"\n✓ Generated {len(generated_files)} test image pairs")
    print(f"✓ Test manifest saved: {manifest_path}")
    print(f"✓ Inference script saved: {inference_script_path}")
    print(f"\n📋 Next steps:")
    print(f"1. Run the inference script to test your model")
    print(f"2. Compare predictions with expected outputs in the manifest")
    print(f"3. Analyze model performance on different sensor combinations")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test image generator for 32x32 greyscale sensor classification model.')
    parser.add_argument('--output_dir', type=str, default='test_images_32x32',
                        help='Directory to save generated test images.')
    parser.add_argument('--mode', type=str, choices=['interactive', 'batch'], default='interactive',
                        help='Generation mode: interactive or batch.')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SENSOR TEST IMAGE GENERATOR (32x32 Format)")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: 32x32 pixels (greyscale)")
    print(f"Mode: {args.mode}")
    print(f"Compatible with trained model input format")
    
    if args.mode == 'interactive':
        interactive_mode(args.output_dir)
    elif args.mode == 'batch':
        batch_mode(args.output_dir)
