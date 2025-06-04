"""
Script untuk membuat dataset kecil dari PlantVillage
Mengambil sample dari setiap kelas untuk testing local
"""

import os
import shutil
import random
from pathlib import Path

def create_small_dataset(source_path="dataset/raw", 
                        target_path="dataset/small", 
                        samples_per_class=50,
                        max_classes=10):
    """
    Buat dataset kecil dengan mengambil sample dari dataset asli
    
    Args:
        source_path: Path dataset asli
        target_path: Path dataset kecil
        samples_per_class: Jumlah gambar per kelas
        max_classes: Maksimal jumlah kelas
    """
    
    print(f"🔄 Membuat dataset kecil...")
    print(f"📁 Source: {source_path}")
    print(f"📁 Target: {target_path}")
    print(f"🖼️  Sample per kelas: {samples_per_class}")
    print(f"📊 Max kelas: {max_classes}")
    
    # Buat folder target
    os.makedirs(target_path, exist_ok=True)
    
    # Hapus isi folder target jika ada
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    os.makedirs(target_path)
    
    # Get semua kelas
    all_classes = [d for d in os.listdir(source_path) 
                   if os.path.isdir(os.path.join(source_path, d))]
    
    # Pilih kelas secara random
    selected_classes = random.sample(all_classes, min(max_classes, len(all_classes)))
    
    total_images = 0
    
    print(f"\n📋 Kelas yang dipilih:")
    for i, class_name in enumerate(selected_classes, 1):
        print(f"  {i}. {class_name}")
        
        source_class_path = os.path.join(source_path, class_name)
        target_class_path = os.path.join(target_path, class_name)
        
        # Buat folder kelas di target
        os.makedirs(target_class_path)
        
        # Get semua gambar dalam kelas
        image_files = [f for f in os.listdir(source_class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Pilih sample secara random
        selected_images = random.sample(image_files, 
                                      min(samples_per_class, len(image_files)))
        
        # Copy gambar ke target
        for img_file in selected_images:
            source_img = os.path.join(source_class_path, img_file)
            target_img = os.path.join(target_class_path, img_file)
            shutil.copy2(source_img, target_img)
        
        total_images += len(selected_images)
        print(f"    ✅ Copied {len(selected_images)} images")
    
    print(f"\n🎉 Dataset kecil berhasil dibuat!")
    print(f"📊 Total: {total_images} gambar dari {len(selected_classes)} kelas")
    print(f"📁 Lokasi: {target_path}")
    
    return target_path, len(selected_classes), total_images

def main():
    """
    Main function
    """
    print("="*60)
    print("🌱 PLANT DISEASE - SMALL DATASET CREATOR")
    print("="*60)
    
    # Cek apakah dataset asli ada
    source_path = "dataset/raw"
    if not os.path.exists(source_path):
        print(f"❌ Dataset asli tidak ditemukan: {source_path}")
        return
    
    # Buat dataset kecil
    target_path, num_classes, total_images = create_small_dataset(
        source_path="dataset/raw",
        target_path="dataset/small", 
        samples_per_class=500,  # 500 gambar per kelas
        max_classes=25          # Maksimal 25 kelas
    )
    
    print(f"\n💡 Langkah selanjutnya:")
    print(f"1. Edit preprocessing.py:")
    print(f"   dataset_path = 'dataset/small'  # Ganti dari 'dataset/raw'")
    print(f"2. Jalankan: python preprocessing.py")
    print(f"3. Jalankan: python train_model.py")
    
    print("="*60)

if __name__ == "__main__":
    # Set random seed untuk reproducibility
    random.seed(42)
    main()