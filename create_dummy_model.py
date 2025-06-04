import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import os

def create_dummy_leaf_model():
    """
    Membuat model dummy untuk klasifikasi daun
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')  # Asumsi 10 kelas daun
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

if __name__ == "__main__":
    # Buat direktori model jika belum ada
    if not os.path.exists('model'):
        os.makedirs('model')
    
    # Buat dan simpan model dummy
    model = create_dummy_leaf_model()
    model.save('model/leaf_model.h5')
    
    print("✅ Model dummy berhasil dibuat dan disimpan!")
    print("📁 File: model/leaf_model.h5")
    print("⚠️  Model ini hanya untuk testing, belum dilatih dengan data!")