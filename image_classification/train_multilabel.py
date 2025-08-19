"""
Custom training script for multi-label sensor classification.
"""
import numpy as np
import tensorflow as tf
from PIL import Image
import json
import os
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from datetime import datetime

# Add the src directory to path
import sys
sys.path.append('src')

from models.stmnist_multilabel import get_stmnist_multilabel


def load_multilabel_dataset(dataset_dir="./datasets/sensor_data_multilabel", validation_split=0.25):
    """Load the multi-label dataset."""
    manifest_path = os.path.join(dataset_dir, "dataset_manifest.json")
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    print(f"Loading dataset: {manifest['num_samples']} samples, {manifest['num_classes']} classes")
    
    # Load all images and labels
    images = []
    labels = []
    
    for sample in manifest['samples']:
        # Load image and resize to 32x32
        img = Image.open(sample['image_path']).convert('L')
        img_resized = img.resize((32, 32), Image.NEAREST)
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
        
        images.append(img_array)
        labels.append(np.array(sample['label_vector'], dtype=np.float32))
    
    images = np.array(images)
    labels = np.array(labels)
    
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image range: {images.min():.3f} to {images.max():.3f}")
    print(f"Label statistics: min={labels.min()}, max={labels.max()}, mean={labels.mean():.3f}")
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=validation_split, random_state=42, stratify=None
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    
    return (X_train, y_train), (X_val, y_val), manifest


def create_model(input_shape=(32, 32, 1), num_classes=8, dropout=0.3):
    """Create the model for multi-label classification."""
    model = get_stmnist_multilabel(
        input_shape=input_shape,
        num_classes=num_classes,
        dropout=dropout
    )
    
    # Compile with appropriate loss and metrics for multi-label classification
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )
    
    return model


def create_callbacks(output_dir):
    """Create training callbacks."""
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'best_model_multilabel.h5'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(output_dir, 'training_log.csv'),
            append=False
        )
    ]
    
    return callbacks


def evaluate_multilabel_model(model, X_test, y_test, threshold=0.5):
    """Evaluate multi-label model performance."""
    predictions = model.predict(X_test)
    predictions_binary = (predictions > threshold).astype(int)
    
    # Calculate per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, predictions_binary, average=None, zero_division=0
    )
    
    # Calculate overall metrics
    exact_match_ratio = accuracy_score(y_test, predictions_binary)
    
    print("\\nEvaluation Results:")
    print(f"Exact Match Ratio: {exact_match_ratio:.4f}")
    print("\\nPer-class metrics:")
    
    class_names = ['high_body_temperature', 'muscle_fatigue', 'emg_fatigue', 'poor_posture', 
                   'high_env_temperature', 'high_stress', 'eye_fatigue', 'normal']
    
    for i, class_name in enumerate(class_names):
        print(f"  {class_name:20}: Precision={precision[i]:.3f}, Recall={recall[i]:.3f}, F1={f1[i]:.3f}, Support={support[i]}")
    
    return exact_match_ratio, predictions


def plot_training_history(history, output_dir):
    """Plot training history."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Training Precision')
    axes[1, 0].plot(history.history['val_precision'], label='Validation Precision')
    axes[1, 0].set_title('Model Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Training Recall')
    axes[1, 1].plot(history.history['val_recall'], label='Validation Recall')
    axes[1, 1].set_title('Model Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main training function."""
    # Create output directory
    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    output_dir = f"experiments_outputs_multilabel/{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting multi-label sensor classification training")
    print(f"Output directory: {output_dir}")
    print("=" * 60)
    
    # Load dataset
    (X_train, y_train), (X_val, y_val), manifest = load_multilabel_dataset()
    
    # Create model
    print("\\nCreating model...")
    model = create_model()
    model.summary()
    
    # Create callbacks
    callbacks = create_callbacks(output_dir)
    
    # Train model
    print("\\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save(os.path.join(output_dir, 'final_model_multilabel.h5'))
    
    # Evaluate model
    print("\\nEvaluating model...")
    exact_match_ratio, predictions = evaluate_multilabel_model(model, X_val, y_val)
    
    # Plot training history
    plot_training_history(history, output_dir)
    
    # Save results
    results = {
        'exact_match_ratio': float(exact_match_ratio),
        'dataset_info': {
            'total_samples': manifest['num_samples'],
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'num_classes': manifest['num_classes'],
            'class_names': manifest['class_names']
        },
        'model_config': {
            'input_shape': [32, 32, 1],
            'num_classes': 8,
            'dropout': 0.3,
            'learning_rate': 0.0001,
            'batch_size': 16,
            'epochs': len(history.history['loss'])
        }
    }
    
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\nTraining completed!")
    print(f"Results saved to: {output_dir}")
    print(f"Best model saved as: {os.path.join(output_dir, 'best_model_multilabel.h5')}")
    print(f"Final exact match ratio: {exact_match_ratio:.4f}")


if __name__ == "__main__":
    main()
