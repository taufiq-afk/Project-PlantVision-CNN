"""
Main preprocessing script untuk PlantVillage dataset
Implementasi 4 teknik preprocessing:
1. Resize
2. Histogram Equalization  
3. Gaussian Blur
4. Normalization
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataset_loader import DatasetLoader
from image_preprocessor import ImagePreprocessor
import json
import time

def main():
    """
    Main function untuk menjalankan preprocessing
    """
    print("="*60)
    print("PLANT VILLAGE DATASET PREPROCESSING")
    print("="*60)
    
    # Inisialisasi
    dataset_path = "dataset/raw"
    processed_path = "processed_data"
    
    # Cek apakah dataset folder ada
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset folder tidak ditemukan: {dataset_path}")
        print("Pastikan dataset PlantVillage sudah diextract ke folder 'dataset/raw'")
        return
    
    # Inisialisasi loader dan preprocessor
    loader = DatasetLoader("dataset/small", processed_path)
    preprocessor = ImagePreprocessor(target_size=(256, 256))
    
    # 1. Tampilkan informasi dataset
    print("\n📊 INFORMASI DATASET")
    print("-" * 40)
    dataset_info = loader.get_dataset_info()
    print(f"Total gambar: {dataset_info['total_images']:,}")
    print(f"Jumlah kelas: {dataset_info['num_classes']}")
    
    # Tampilkan distribusi kelas (top 10)
    print("\n🏷️  Top 10 Kelas dengan Jumlah Gambar Terbanyak:")
    sorted_classes = sorted(dataset_info['class_distribution'].items(), 
                          key=lambda x: x[1], reverse=True)[:10]
    for class_name, count in sorted_classes:
        print(f"  • {class_name}: {count:,} gambar")
    
    # 2. Load dataset
    print(f"\n📥 LOADING DATASET")
    print("-" * 40)
    start_time = time.time()
    
    try:
        X, y, class_info = loader.load_raw_dataset()
        load_time = time.time() - start_time
        print(f"✅ Dataset berhasil dimuat dalam {load_time:.2f} detik")
        print(f"Shape gambar: {X.shape}")
        print(f"Shape label: {y.shape}")
        
    except Exception as e:
        print(f"❌ Error loading dataset: {str(e)}")
        return
    
    # 3. Split dataset
    print(f"\n🔄 SPLITTING DATASET")
    print("-" * 40)
    X_train, X_val, X_test, y_train, y_val, y_test = loader.split_dataset(
        X, y, test_size=0.15, val_size=0.15, random_state=42
    )
    
    # 4. Preprocessing
    print(f"\n⚙️  PREPROCESSING IMAGES")
    print("-" * 40)
    print("Teknik yang digunakan:")
    print("  1. ✂️  Resize ke 128x128")
    print("  2. 📈 Histogram Equalization (CLAHE)") 
    print("  3. 🌫️  Gaussian Blur (noise reduction)")
    print("  4. 🔢 Normalization (0-1 range)")
    
    start_time = time.time()
    
    # Preprocess setiap split
    print("\n🚀 Memproses data training...")
    X_train_processed = preprocessor.preprocess_batch(X_train, 
                                                     apply_blur=True, 
                                                     apply_hist_eq=True)
    
    print("🚀 Memproses data validation...")
    X_val_processed = preprocessor.preprocess_batch(X_val, 
                                                   apply_blur=True, 
                                                   apply_hist_eq=True)
    
    print("🚀 Memproses data testing...")
    X_test_processed = preprocessor.preprocess_batch(X_test, 
                                                    apply_blur=True, 
                                                    apply_hist_eq=True)
    
    preprocessing_time = time.time() - start_time
    print(f"✅ Preprocessing selesai dalam {preprocessing_time:.2f} detik")
    
    # 5. Simpan data yang sudah dipreprocess
    print(f"\n💾 MENYIMPAN DATA PREPROCESSED")
    print("-" * 40)
    preprocessor.save_preprocessed_data(
        X_train_processed, X_val_processed, X_test_processed,
        y_train, y_val, y_test, processed_path
    )
    
    # 6. Statistik akhir
    print(f"\n📊 RINGKASAN PREPROCESSING")
    print("-" * 40)
    print(f"Dataset original:")
    print(f"  • Total gambar: {len(X):,}")
    print(f"  • Ukuran rata-rata: {np.mean([img.shape[:2] for img in X], axis=0)}")
    
    print(f"\nDataset setelah preprocessing:")
    print(f"  • Train: {X_train_processed.shape}")
    print(f"  • Validation: {X_val_processed.shape}")
    print(f"  • Test: {X_test_processed.shape}")
    print(f"  • Ukuran standar: {preprocessor.target_size}")
    print(f"  • Range nilai: [0.0, 1.0]")
    
    # 7. Visualisasi contoh preprocessing
    print(f"\n🖼️  MEMBUAT VISUALISASI CONTOH")
    print("-" * 40)
    create_preprocessing_visualization(X[0], preprocessor, processed_path)
    
    print(f"\n🎉 PREPROCESSING SELESAI!")
    print(f"Data tersimpan di folder: {processed_path}")
    print("="*60)

def create_preprocessing_visualization(sample_image, preprocessor, save_path):
    """
    Buat visualisasi contoh preprocessing steps
    """
    try:
        fig = preprocessor.visualize_preprocessing_steps(sample_image)
        
        # Simpan visualisasi
        viz_path = os.path.join(save_path, 'preprocessing_visualization.png')
        fig.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✅ Visualisasi preprocessing tersimpan: {viz_path}")
        
    except Exception as e:
        print(f"⚠️  Tidak dapat membuat visualisasi: {str(e)}")

def check_preprocessed_data():
    """
    Function untuk cek data yang sudah dipreprocess
    """
    processed_path = "processed_data"
    
    if not os.path.exists(processed_path):
        print("❌ Folder processed_data tidak ditemukan")
        return
    
    required_files = [
        'X_train.npy', 'X_val.npy', 'X_test.npy',
        'y_train.npy', 'y_val.npy', 'y_test.npy',
        'class_info.json', 'preprocessing_params.pkl'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(processed_path, file)):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ File yang hilang: {missing_files}")
        print("Jalankan preprocessing terlebih dahulu")
        return False
    
    # Load dan tampilkan info
    preprocessor = ImagePreprocessor()
    X_train, X_val, X_test, y_train, y_val, y_test, params = preprocessor.load_preprocessed_data(processed_path)
    
    print("✅ Data preprocessed tersedia:")
    print(f"  • Train: {X_train.shape}")
    print(f"  • Val: {X_val.shape}")
    print(f"  • Test: {X_test.shape}")
    print(f"  • Target size: {params['target_size']}")
    
    return True

if __name__ == "__main__":
    # Cek command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--check":
            print("🔍 Mengecek data preprocessed...")
            check_preprocessed_data()
        elif sys.argv[1] == "--help":
            print("Usage:")
            print("  python preprocessing.py          # Jalankan preprocessing")
            print("  python preprocessing.py --check  # Cek data preprocessed")
            print("  python preprocessing.py --help   # Tampilkan help")
        else:
            print("❌ Argument tidak dikenal. Gunakan --help untuk bantuan")
    else:
        # Jalankan preprocessing
        main()