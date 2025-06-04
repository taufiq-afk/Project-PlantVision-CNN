"""
Utility functions untuk dataset management
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import cv2
import random

def analyze_dataset_distribution(dataset_path="dataset/raw"):
    """
    Analisis distribusi dataset PlantVillage
    """
    class_counts = {}
    total_images = 0
    image_sizes = []
    
    print("🔍 Menganalisis distribusi dataset...")
    
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        
        if os.path.isdir(class_path):
            count = 0
            for image_file in os.listdir(class_path):
                if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    count += 1
                    
                    # Sample beberapa gambar untuk analisis ukuran
                    if count <= 5:  # Sample 5 gambar per kelas
                        img_path = os.path.join(class_path, image_file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            image_sizes.append(img.shape[:2])  # (height, width)
            
            class_counts[class_folder] = count
            total_images += count
    
    # Statistik
    stats = {
        'total_images': total_images,
        'num_classes': len(class_counts),
        'class_distribution': class_counts,
        'avg_images_per_class': total_images / len(class_counts),
        'min_images_per_class': min(class_counts.values()),
        'max_images_per_class': max(class_counts.values()),
    }
    
    if image_sizes:
        heights, widths = zip(*image_sizes)
        stats['image_size_stats'] = {
            'avg_height': np.mean(heights),
            'avg_width': np.mean(widths),
            'min_height': min(heights),
            'max_height': max(heights),
            'min_width': min(widths),
            'max_width': max(widths)
        }
    
    return stats

def plot_class_distribution(class_counts, save_path=None, top_n=20):
    """
    Plot distribusi kelas
    """
    # Sort berdasarkan jumlah
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Ambil top N
    if len(sorted_classes) > top_n:
        top_classes = sorted_classes[:top_n]
        other_count = sum([count for _, count in sorted_classes[top_n:]])
        top_classes.append(('Others', other_count))
    else:
        top_classes = sorted_classes
    
    # Plot
    plt.figure(figsize=(15, 8))
    classes, counts = zip(*top_classes)
    
    # Clean class names (remove prefix)
    clean_classes = []
    for cls in classes:
        if '___' in cls:
            clean_classes.append(cls.split('___')[1].replace('_', ' '))
        else:
            clean_classes.append(cls.replace('_', ' '))
    
    plt.bar(range(len(clean_classes)), counts, color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Kelas Penyakit', fontsize=12)
    plt.ylabel('Jumlah Gambar', fontsize=12)
    plt.title(f'Distribusi Dataset PlantVillage (Top {top_n} Kelas)', fontsize=14, fontweight='bold')
    
    plt.xticks(range(len(clean_classes)), clean_classes, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Tambahkan nilai di atas bar
    for i, count in enumerate(counts):
        plt.text(i, count + max(counts)*0.01, str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Plot distribusi tersimpan: {save_path}")
    
    return plt.gcf()

def sample_images_from_classes(dataset_path="dataset/raw", num_classes=6, num_samples=3):
    """
    Ambil sample gambar dari beberapa kelas untuk visualisasi
    """
    classes = [d for d in os.listdir(dataset_path) 
              if os.path.isdir(os.path.join(dataset_path, d))]
    
    # Pilih kelas secara random
    selected_classes = random.sample(classes, min(num_classes, len(classes)))
    
    samples = {}
    
    for class_name in selected_classes:
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Pilih sample secara random
        selected_files = random.sample(image_files, min(num_samples, len(image_files)))
        
        class_samples = []
        for img_file in selected_files:
            img_path = os.path.join(class_path, img_file)
            img = cv2.imrea