"""
Interactive script to generate test images with user-selected sensor states.
Allows runtime selection of which sensors should be red (alert) or blue (normal).
Perfect for testing CNN models with custom sensor combinations.
"""
import os
import argparse
from PIL import Image, ImageDraw
import sys

def display_sensor_grid():
    """Display the 16x2 grid layout with sensor positions."""
    print("\n" + "="*60)
    print("SENSOR GRID LAYOUT (16x2 Grid, Greyscale)")
    print("="*60)
    print("Grid Position:  Sensor Name:")
    print("Position 0     high_body_temperature    (pixels 0-1, rows 0-1)")
    print("Position 1     muscle_fatigue           (pixels 2-3, rows 0-1)") 
    print("Position 2     emg_fatigue              (pixels 4-5, rows 0-1)")
    print("Position 3     poor_posture             (pixels 6-7, rows 0-1)")
    print("Position 4     high_env_temperature     (pixels 8-9, rows 0-1)")
    print("Position 5     high_stress              (pixels 10-11, rows 0-1)")
    print("Position 6     eye_fatigue              (pixels 12-13, rows 0-1)")
    print("Position 7     normal_state             (pixels 14-15, rows 0-1)")
    print("="*60)
    print("Visual Grid Layout (16x2, each position is 2x2 pixels):")
    print("┌──┬──┬──┬──┬──┬──┬──┬──┐")
    print("│ 0│ 1│ 2│ 3│ 4│ 5│ 6│ 7│ <- Row 0-1") 
    print("└──┴──┴──┴──┴──┴──┴──┴──┘")
    print("Colors: BLACK = Normal/Inactive, WHITE = Alert/Active")
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

def create_test_image(sensor_states, image_size=16):
    """Create test image based on sensor states in 16x2 greyscale format."""
    # Create 16x2 greyscale image with black background
    img = Image.new('L', (16, 2), 0)  # 'L' mode for greyscale, 0 = black
    draw = ImageDraw.Draw(img)
    
    # Fill sensor positions based on user selection
    for position in range(7):
        x0 = position * 2  # Each sensor gets 2 pixels width
        y0 = 0             # All sensors are in the 2-row height
        x1 = x0 + 2        # 2x2 square
        y1 = 2             # Full height
        bbox = (x0, y0, x1, y1)
        
        if sensor_states.get(position, 'black') == 'white':
            draw.rectangle(bbox, fill=255)  # White = alert
        else:
            draw.rectangle(bbox, fill=0)    # Black = normal
    
    # Handle normal state indicator at position 7
    all_sensors_normal = all(sensor_states.get(i, 'black') == 'black' for i in range(7))
    if all_sensors_normal:
        # If all sensors are normal (black), set normal indicator to white
        x0 = 7 * 2  # Position 7: (14,0) to (15,1)
        y0 = 0
        x1 = x0 + 2
        y1 = 2
        bbox = (x0, y0, x1, y1)
        draw.rectangle(bbox, fill=255)  # White = normal state active
    
    return img

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
    """Generate multi-label binary vector for CNN testing."""
    # 7 sensors + 1 normal indicator = 8 total outputs
    vector = []
    
    # First 7 positions for sensors
    for position in range(7):
        vector.append(1 if sensor_states.get(position, 'black') == 'white' else 0)
    
    # 8th position for normal state (1 if all sensors are normal, 0 otherwise)
    all_normal = all(sensor_states.get(i, 'black') == 'black' for i in range(7))
    vector.append(1 if all_normal else 0)
    
    return vector

def save_test_image(img, sensor_states, output_dir):
    """Save the test image with descriptive filename."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename based on active sensors
    white_positions = [str(pos) for pos in range(7) if sensor_states.get(pos, 'black') == 'white']
    
    if not white_positions:
        filename = "test_all_normal.png"
    elif len(white_positions) == 7:
        filename = "test_all_alert.png"
    else:
        filename = f"test_sensors_{'_'.join(white_positions)}.png"
    
    filepath = os.path.join(output_dir, filename)
    img.save(filepath)
    
    return filepath

def interactive_mode(output_dir, image_size):
    """Run interactive mode for single image generation."""
    display_sensor_grid()
    
    while True:
        print("\n" + "="*60)
        print("INTERACTIVE TEST IMAGE GENERATOR")
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
        
        # Generate image
        img = create_test_image(sensor_states, 16)  # Fixed 16x2 size
        
        # Save image
        filepath = save_test_image(img, sensor_states, output_dir)
        
        # Generate multi-label vector for CNN testing
        ml_vector = generate_multilabel_vector(sensor_states)
        
        print(f"\n✓ Image saved: {filepath}")
        print(f"✓ Image size: 16x2 pixels (greyscale)")
        print(f"✓ Multi-label vector: {ml_vector}")
        print(f"✓ Binary representation: {''.join(map(str, ml_vector))}")
        
        # Ask if user wants to generate another
        continue_choice = input("\nGenerate another image? (y/n): ").lower().strip()
        if continue_choice not in ['y', 'yes']:
            break

def batch_mode(output_dir, image_size, predefined_tests):
    """Generate a batch of predefined test cases."""
    print("\n" + "="*60)
    print("BATCH MODE: Generating predefined test cases")
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
        
        # Multiple sensor combinations
        ({0: 'white', 4: 'white'}, "heat_related"),
        ({1: 'white', 3: 'white'}, "muscle_posture"),
        ({5: 'white', 6: 'white'}, "stress_eye"),
        ({1: 'white', 2: 'white', 3: 'white'}, "physical_strain"),
        ({5: 'white', 6: 'white', 1: 'white'}, "work_overload"),
        
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
        
        # Create image
        img = create_test_image(sensor_states, 16)  # Fixed 16x2 size
        
        # Save with predefined name
        filename = f"test_{test_name}.png"
        filepath = os.path.join(output_dir, filename)
        img.save(filepath)
        
        # Generate info
        ml_vector = generate_multilabel_vector(sensor_states)
        white_count = sum(ml_vector[:-1])  # Exclude normal indicator from count
        
        print(f"  ✓ Saved: {filename}")
        print(f"  ✓ Active sensors: {white_count}/7")
        print(f"  ✓ Multi-label: {ml_vector}")
        
        generated_files.append((filepath, ml_vector, test_name))
    
    # Save test manifest
    manifest_path = os.path.join(output_dir, "test_manifest.csv")
    with open(manifest_path, 'w') as f:
        f.write("filename,test_name,multi_label_vector,active_sensors,normal_indicator\n")
        for filepath, ml_vector, test_name in generated_files:
            filename = os.path.basename(filepath)
            vector_str = ','.join(map(str, ml_vector))
            active_sensors = sum(ml_vector[:-1])  # Exclude normal indicator
            normal_indicator = ml_vector[-1]
            f.write(f"{filename},{test_name},\"{vector_str}\",{active_sensors},{normal_indicator}\n")
    
    print(f"\n✓ Generated {len(generated_files)} test images")
    print(f"✓ Test manifest saved: {manifest_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Interactive test image generator for 16x2 greyscale sensor data.')
    parser.add_argument('--output_dir', type=str, default='test_images',
                        help='Directory to save generated test images.')
    parser.add_argument('--image_size', type=int, default=16,
                        help='Width of each image (height is fixed at 2).')
    parser.add_argument('--mode', type=str, choices=['interactive', 'batch'], default='interactive',
                        help='Generation mode: interactive or batch.')
    parser.add_argument('--batch_tests', action='store_true',
                        help='Generate predefined test cases in batch mode.')
    
    args = parser.parse_args()
    
    print("="*60)
    print("SENSOR TEST IMAGE GENERATOR (16x2 Greyscale)")
    print("="*60)
    print(f"Output directory: {args.output_dir}")
    print(f"Image size: 16x2 pixels (greyscale)")
    print(f"Mode: {args.mode}")
    
    if args.mode == 'interactive':
        interactive_mode(args.output_dir, args.image_size)
    elif args.mode == 'batch':
        batch_mode(args.output_dir, args.image_size, args.batch_tests)
