"""
Pipeline script untuk menjalankan seluruh proses training
dari preprocessing hingga evaluasi model
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import time

def check_dataset():
    """
    Cek apakah dataset PlantVillage tersedia
    """
    dataset_path = "dataset/raw"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset folder tidak ditemukan: {dataset_path}")
        print("📥 Download dataset PlantVillage dari Kaggle:")
        print("   https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset")
        print("📁 Extract ke folder 'dataset/raw/'")
        return False
    
    # Hitung jumlah folder kelas
    class_folders = [d for d in os.listdir(dataset_path) 
                    if os.path.isdir(os.path.join(dataset_path, d))]
    
    if len(class_folders) < 10:  # Minimal harus ada beberapa kelas
        print(f"⚠️  Hanya ditemukan {len(class_folders)} kelas dalam dataset")
        print("📁 Pastikan dataset sudah diekstrak dengan benar")
        return False
    
    total_images = 0
    for folder in class_folders[:5]:  # Sample 5 folder untuk hitung gambar
        folder_path = os.path.join(dataset_path, folder)
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        total_images += len(images)
    
    print(f"✅ Dataset ditemukan:")
    print(f"  📂 {len(class_folders)} kelas")
    print(f"  🖼️  ~{total_images * len(class_folders) // 5:,} gambar (estimasi)")
    
    return True

def check_dependencies():
    """
    Cek apakah semua dependencies sudah terinstall
    """
    required_packages = [
        'tensorflow', 'opencv-python', 'numpy', 'matplotlib', 
        'scikit-learn', 'flask', 'tqdm', 'seaborn'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("💡 Install dengan: pip install -r requirements.txt")
        return False
    
    print("✅ Semua dependencies tersedia")
    return True

def run_preprocessing():
    """
    Jalankan preprocessing
    """
    print("\n" + "="*60)
    print("🔄 TAHAP 1: PREPROCESSING")
    print("="*60)
    
    try:
        result = subprocess.run([sys.executable, 'preprocessing.py'], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ Preprocessing berhasil!")
            return True
        else:
            print("❌ Preprocessing gagal!")
            return False
            
    except Exception as e:
        print(f"❌ Error menjalankan preprocessing: {e}")
        return False

def run_training():
    """
    Jalankan training model
    """
    print("\n" + "="*60)
    print("🚀 TAHAP 2: TRAINING MODEL")
    print("="*60)
    
    try:
        result = subprocess.run([sys.executable, 'train_model.py'], 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("✅ Training berhasil!")
            return True
        else:
            print("❌ Training gagal!")
            return False
            
    except Exception as e:
        print(f"❌ Error menjalankan training: {e}")
        return False

def test_web_app():
    """
    Test web application
    """
    print("\n" + "="*60)
    print("🌐 TAHAP 3: TEST WEB APPLICATION")
    print("="*60)
    
    # Cek apakah model sudah ada
    model_path = "model/leaf_model.h5"
    if not os.path.exists(model_path):
        print(f"❌ Model tidak ditemukan: {model_path}")
        return False
    
    # Cek processed data
    processed_files = ['class_info.json', 'preprocessing_params.pkl']
    for file in processed_files:
        if not os.path.exists(f"processed_data/{file}"):
            print(f"❌ File tidak ditemukan: processed_data/{file}")
            return False
    
    print("✅ Semua file model tersedia")
    print("🌐 Web app siap dijalankan!")
    print("💡 Jalankan: python app.py")
    
    return True

def show_summary():
    """
    Tampilkan ringkasan hasil
    """
    print("\n" + "="*60)
    print("📊 RINGKASAN PIPELINE")
    print("="*60)
    
    # Cek file yang dihasilkan
    files_to_check = {
        "📊 Preprocessed Data": [
            "processed_data/X_train.npy",
            "processed_data/X_val.npy", 
            "processed_data/X_test.npy",
            "processed_data/class_info.json"
        ],
        "🤖 Model Files": [
            "model/leaf_model.h5",
            "model/best_model.h5",
            "model/model_info.json",
            "model/training_history.png"
        ]
    }
    
    for category, files in files_to_check.items():
        print(f"\n{category}:")
        for file_path in files:
            if os.path.exists(file_path):
                size = os.path.getsize(file_path)
                if size > 1024*1024:  # > 1MB
                    size_str = f"{size/(1024*1024):.1f}MB"
                elif size > 1024:  # > 1KB
                    size_str = f"{size/1024:.1f}KB"
                else:
                    size_str = f"{size}B"
                print(f"  ✅ {file_path} ({size_str})")
            else:
                print(f"  ❌ {file_path} (tidak ditemukan)")
    
    print(f"\n🎉 Pipeline selesai!")
    print(f"🌐 Jalankan web app: python app.py")
    print(f"📱 Akses di: http://localhost:5000")

def main():
    parser = argparse.ArgumentParser(description='Plant Disease Classification Pipeline')
    parser.add_argument('--step', choices=['all', 'preprocess', 'train', 'test'], 
                       default='all', help='Pilih tahap yang ingin dijalankan')
    parser.add_argument('--skip-checks', action='store_true', 
                       help='Skip pengecekan dataset dan dependencies')
    
    args = parser.parse_args()
    
    print("🌱 PLANT DISEASE CLASSIFICATION PIPELINE")
    print("="*60)
    
    if not args.skip_checks:
        # 1. Cek dataset
        if not check_dataset():
            print("\n❌ Pipeline dibatalkan karena dataset tidak tersedia")
            return
        
        # 2. Cek dependencies  
        if not check_dependencies():
            print("\n❌ Pipeline dibatalkan karena dependencies tidak lengkap")
            return
    
    success = True
    
    # Jalankan tahapan sesuai pilihan
    if args.step in ['all', 'preprocess']:
        if not run_preprocessing():
            success = False
    
    if success and args.step in ['all', 'train']:
        if not run_training():
            success = False
    
    if success and args.step in ['all', 'test']:
        if not test_web_app():
            success = False
    
    # Tampilkan ringkasan
    if args.step == 'all':
        show_summary()
    
    if success:
        print(f"\n🎉 Tahap '{args.step}' berhasil diselesaikan!")
    else:
        print(f"\n❌ Tahap '{args.step}' gagal!")

if __name__ == "__main__":
    main()