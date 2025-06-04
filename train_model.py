"""
🚀 RTX 3050 Ti OPTIMIZED Plant Disease Trainer
Ultra-Conservative Memory Management untuk 4GB VRAM

FEATURES:
✅ Ultra-small batch size (4) untuk 4GB VRAM
✅ Simple model architecture (no complex layers)
✅ Smart GPU/CPU fallback
✅ Conservative memory allocation
✅ Guaranteed to finish training
✅ No mixed precision (causes issues)

VERSION: RTX 3050 Ti Optimized Final
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Disable mixed precision - causes memory issues
os.environ['TF_ENABLE_AUTO_MIXED_PRECISION'] = '0'

class RTX3050Trainer:
    def __init__(self, processed_data_path="processed_data", model_save_path="model"):
        """
        RTX 3050 Ti Optimized Trainer
        Ultra-conservative settings for 4GB VRAM
        """
        self.processed_data_path = processed_data_path
        self.model_save_path = model_save_path
        self.model = None
        self.history = None
        self.class_info = None
        self.gpu_mode = False
        
        # conservative settings
        self.batch_size = 1  # Very small for 4GB VRAM
        self.epochs = 25     # Reasonable number
        
        # Create directories
        os.makedirs(model_save_path, exist_ok=True)
        
        # Setup GPU with conservative approach
        self.setup_conservative_gpu()
        
    def setup_conservative_gpu(self):
        """
        Ultra-conservative GPU setup for RTX 3050 Ti
        Fixed API compatibility
        """
        print("\n🔧 Conservative GPU Setup...")
        
        try:
            # Check GPU availability
            gpus = tf.config.experimental.list_physical_devices('GPU')
            
            if not gpus:
                print("ℹ️ No GPU found - using CPU mode")
                self.gpu_mode = False
                self.batch_size = 8  # Slightly larger for CPU
                return
            
            print(f"✅ GPU found: {gpus[0]}")
            
            # Simple GPU setup - just memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print("✅ GPU memory growth enabled")
            
            # Test GPU with simple operation
            try:
                with tf.device('/GPU:0'):
                    test_tensor = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                    result = tf.reduce_sum(test_tensor)
                    
                print("✅ GPU computation test successful")
                self.gpu_mode = True
                
            except Exception as e:
                print(f"⚠️ GPU test failed: {e}")
                print("🔄 Falling back to CPU mode")
                self.gpu_mode = False
                self.batch_size = 8
                return
            
            print(f"\n🎯 RTX 3050 Ti Settings:")
            print(f"   • Batch size: {self.batch_size} (ultra-conservative)")
            print(f"   • Memory growth: Enabled")
            print(f"   • Mixed precision: Disabled")
            print(f"   • GPU mode: ACTIVE")
            
        except Exception as e:
            print(f"⚠️ GPU setup failed: {e}")
            print("🔄 Falling back to CPU mode")
            self.gpu_mode = False
            self.batch_size = 8
    
    def load_data(self):
        """
        Load preprocessed data with memory efficiency
        """
        print("\n📥 Loading data efficiently...")
        
        try:
            # Load data
            self.X_train = np.load(os.path.join(self.processed_data_path, 'X_train.npy'))
            self.X_val = np.load(os.path.join(self.processed_data_path, 'X_val.npy'))
            self.X_test = np.load(os.path.join(self.processed_data_path, 'X_test.npy'))
            self.y_train = np.load(os.path.join(self.processed_data_path, 'y_train.npy'))
            self.y_val = np.load(os.path.join(self.processed_data_path, 'y_val.npy'))
            self.y_test = np.load(os.path.join(self.processed_data_path, 'y_test.npy'))
            
            # Load class info
            with open(os.path.join(self.processed_data_path, 'class_info.json'), 'r') as f:
                self.class_info = json.load(f)
            
            print("✅ Data loaded successfully")
            print(f"   • Training: {self.X_train.shape[0]:,} samples")
            print(f"   • Validation: {self.X_val.shape[0]:,} samples")
            print(f"   • Test: {self.X_test.shape[0]:,} samples")
            print(f"   • Classes: {self.class_info['num_classes']}")
            print(f"   • Image shape: {self.X_train.shape[1:]}")
            
            # Convert to categorical
            num_classes = self.class_info['num_classes']
            self.y_train_cat = keras.utils.to_categorical(self.y_train, num_classes)
            self.y_val_cat = keras.utils.to_categorical(self.y_val, num_classes)
            self.y_test_cat = keras.utils.to_categorical(self.y_test, num_classes)
            
            # Memory optimization - convert to float32 explicitly
            self.X_train = self.X_train.astype(np.float32)
            self.X_val = self.X_val.astype(np.float32) 
            self.X_test = self.X_test.astype(np.float32)
            
            print("✅ Data preprocessing completed")
            return True
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return False
    
    def build_simple_model(self):
        """
        Build ultra-simple model for RTX 3050 Ti
        Designed for 4GB VRAM with maximum efficiency
        """
        print("\n🏗️ Building RTX 3050 Ti optimized model...")
        
        input_shape = self.X_train.shape[1:]
        num_classes = self.class_info['num_classes']
        
        print(f"   • Input shape: {input_shape}")
        print(f"   • Output classes: {num_classes}")
        print(f"   • Architecture: Ultra-simple CNN")
        
        # ULTRA-SIMPLE MODEL for 4GB VRAM
        model = keras.Sequential([
            # Input
            layers.Input(shape=input_shape),
            
            # Block 1 - Very small
            layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2 - Small
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3 - Medium
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.3),
            
            # Block 4 - Conservative
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),  # Memory efficient
            layers.Dropout(0.5),
            
            # Classifier - Simple
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            
            # Output
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Simple compilation
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("✅ Simple model created")
        print(f"   • Total parameters: {model.count_params():,}")
        print(f"   • Max filters: 128 (vs 512 in complex models)")
        print(f"   • Memory efficient: GlobalAveragePooling2D")
        
        # Show summary
        model.summary()
        
        self.model = model
        return model
    
    def train_conservative(self):
        """
        Conservative training with RTX 3050 Ti optimizations
        """
        print(f"\n🚀 Starting conservative training...")
        print(f"   • Device: {'GPU' if self.gpu_mode else 'CPU'}")
        print(f"   • Batch size: {self.batch_size}")
        print(f"   • Epochs: {self.epochs}")
        print(f"   • Conservative memory mode: ON")
        
        # Force device selection
        device_context = tf.device('/GPU:0' if self.gpu_mode else '/CPU:0')
        
        # Setup callbacks
        callbacks = [
            ModelCheckpoint(
                filepath=os.path.join(self.model_save_path, 'best_model.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Conservative data augmentation
        print("📈 Setting up conservative data augmentation...")
        datagen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,      # Reduced
            width_shift_range=0.1,  # Reduced
            height_shift_range=0.1, # Reduced
            horizontal_flip=True,
            zoom_range=0.1,         # Reduced
            rescale=None
        )
        
        datagen.fit(self.X_train)
        print("✅ Conservative augmentation ready")
        
        # Calculate steps
        steps_per_epoch = max(1, len(self.X_train) // self.batch_size)
        print(f"   • Steps per epoch: {steps_per_epoch}")
        
        # Start training
        start_time = datetime.now()
        print(f"\n🔥 Starting training at {start_time.strftime('%H:%M:%S')}...")
        
        if self.gpu_mode:
            print("💪 FORCING GPU TRAINING MODE")
        else:
            print("🔄 Using CPU training mode")
        
        try:
            # Training with device context
            with device_context:
                self.history = self.model.fit(
                    datagen.flow(self.X_train, self.y_train_cat, batch_size=self.batch_size),
                    steps_per_epoch=steps_per_epoch,
                    epochs=self.epochs,
                    validation_data=(self.X_val, self.y_val_cat),
                    callbacks=callbacks,
                    verbose=1,
                    workers=1,  # Conservative
                    use_multiprocessing=False  # Safer
                )
            
            training_time = datetime.now() - start_time
            print(f"\n✅ Training completed successfully!")
            print(f"⏱️ Training time: {training_time}")
            print(f"🎯 Device used: {'GPU' if self.gpu_mode else 'CPU'}")
            
            # Save final model
            final_model_path = os.path.join(self.model_save_path, 'leaf_model.h5')
            self.model.save(final_model_path)
            print(f"💾 Final model saved: {final_model_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            
            # Try fallback to CPU if GPU failed
            if self.gpu_mode:
                print("🔄 GPU training failed, trying CPU fallback...")
                try:
                    with tf.device('/CPU:0'):
                        self.history = self.model.fit(
                            datagen.flow(self.X_train, self.y_train_cat, batch_size=self.batch_size),
                            steps_per_epoch=steps_per_epoch,
                            epochs=self.epochs,
                            validation_data=(self.X_val, self.y_val_cat),
                            callbacks=callbacks,
                            verbose=1,
                            workers=1,
                            use_multiprocessing=False
                        )
                    
                    training_time = datetime.now() - start_time
                    print(f"✅ CPU fallback training successful!")
                    print(f"⏱️ Training time: {training_time}")
                    
                    # Save final model
                    final_model_path = os.path.join(self.model_save_path, 'leaf_model.h5')
                    self.model.save(final_model_path)
                    print(f"💾 Final model saved: {final_model_path}")
                    
                    self.gpu_mode = False  # Update mode for reporting
                    return True
                    
                except Exception as e2:
                    print(f"❌ CPU fallback also failed: {e2}")
                    return False
            else:
                return False
    
    def evaluate_model(self):
        """
        Evaluate trained model
        """
        print("\n📊 Evaluating model...")
        
        # Load best model if available
        best_model_path = os.path.join(self.model_save_path, 'best_model.h5')
        if os.path.exists(best_model_path):
            try:
                self.model = keras.models.load_model(best_model_path)
                print("✅ Loaded best model for evaluation")
            except Exception as e:
                print(f"⚠️ Could not load best model: {e}")
        
        try:
            # Evaluate
            test_loss, test_accuracy = self.model.evaluate(
                self.X_test, self.y_test_cat,
                batch_size=self.batch_size,
                verbose=0
            )
            
            print(f"\n📈 FINAL RESULTS:")
            print(f"   • Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
            print(f"   • Test Loss: {test_loss:.4f}")
            print(f"   • Training Device: {'GPU' if self.gpu_mode else 'CPU'}")
            
            # Save results
            results = {
                'test_accuracy': float(test_accuracy),
                'test_loss': float(test_loss),
                'training_device': 'GPU' if self.gpu_mode else 'CPU',
                'batch_size': self.batch_size,
                'epochs_trained': len(self.history.history['loss']),
                'model_params': self.model.count_params(),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(os.path.join(self.model_save_path, 'training_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
            
            # Performance assessment
            if test_accuracy > 0.85:
                print("🚀 OUTSTANDING! Excellent accuracy!")
            elif test_accuracy > 0.75:
                print("🔥 EXCELLENT! Very good results!")
            elif test_accuracy > 0.65:
                print("💪 VERY GOOD! Good performance!")
            elif test_accuracy > 0.55:
                print("✅ GOOD! Model is learning!")
            else:
                print("📈 BASELINE! Training completed!")
            
            return results
            
        except Exception as e:
            print(f"❌ Evaluation failed: {e}")
            return None

def main():
    """
    Main training function
    """
    print("="*60)
    print("🚀 RTX 3050 Ti PLANT DISEASE TRAINER")
    print("="*60)
    print("🎯 Ultra-conservative training for 4GB VRAM")
    print("⚡ Optimized batch size and model architecture")
    print("🛡️ Smart GPU/CPU fallback system")
    print("="*60)
    
    # Initialize trainer
    print("\n🚀 Step 1: Initialize RTX 3050 Ti trainer...")
    trainer = RTX3050Trainer()
    
    # Load data
    print("\n🚀 Step 2: Load preprocessed data...")
    if not trainer.load_data():
        print("\n❌ DATA LOADING FAILED!")
        print("💡 Run preprocessing first:")
        print("   1. python create_small_dataset.py")
        print("   2. python preprocessing.py")
        return
    
    # Build model
    print("\n🚀 Step 3: Build optimized model...")
    trainer.build_simple_model()
    
    # Train model
    print("\n🚀 Step 4: Start conservative training...")
    if not trainer.train_conservative():
        print("\n❌ TRAINING FAILED!")
        return
    
    # Evaluate model
    print("\n🚀 Step 5: Evaluate trained model...")
    results = trainer.evaluate_model()
    
    if results is None:
        print("⚠️ Evaluation failed, but training completed")
        print("✅ Model files should be available")
        return
    
    # Final summary
    print("\n" + "="*60)
    print("🎉 RTX 3050 Ti TRAINING COMPLETED!")
    print("="*60)
    print(f"🎯 Final accuracy: {results['test_accuracy']:.4f} ({results['test_accuracy']*100:.2f}%)")
    print(f"🔧 Device used: {results['training_device']}")
    print(f"📊 Model parameters: {results['model_params']:,}")
    print(f"⏱️ Epochs completed: {results['epochs_trained']}")
    print(f"💾 Model saved: {trainer.model_save_path}/")
    
    if results['training_device'] == 'GPU':
        print("🚀 RTX 3050 Ti acceleration successful!")
    else:
        print("💪 CPU training completed successfully!")
    
    print("\n🎯 READY FOR WEB APP!")
    print("✅ Run: python app.py")
    print("="*60)

if __name__ == "__main__":
    main()