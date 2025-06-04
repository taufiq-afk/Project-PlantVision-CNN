import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import json
from pathlib import Path

class DatasetLoader:
    def __init__(self, dataset_path="dataset/raw", processed_path="processed_data"):
        self.dataset_path = dataset_path
        self.processed_path = processed_path
        self.label_encoder = LabelEncoder()
        
        # Buat folder processed_data jika belum ada
        os.makedirs(processed_path, exist_ok=True)
        
    def load_raw_dataset(self):
        """
        Load dataset PlantVillage dari folder raw
        """
        images = []
        labels = []
        class_names = []
        
        print("Loading dataset from:", self.dataset_path)
        
        # Iterasi melalui setiap folder kelas
        for class_folder in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_folder)
            
            if os.path.isdir(class_path):
                class_names.append(class_folder)
                print(f"Loading class: {class_folder}")
                
                # Load semua gambar dalam folder kelas
                image_count = 0
                for image_file in os.listdir(class_path):
                    if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        image_path = os.path.join(class_path, image_file)
                        
                        # Load gambar
                        image = cv2.imread(image_path)
                        if image is not None:
                            # Convert BGR ke RGB
                            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                            images.append(image)
                            labels.append(class_folder)
                            image_count += 1
                
                print(f"  Loaded {image_count} images")
        
        print(f"Total loaded: {len(images)} images from {len(class_names)} classes")
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        # Simpan informasi kelas
        class_info = {
            'class_names': class_names,
            'num_classes': len(class_names),
            'label_mapping': dict(zip(range(len(class_names)), class_names))
        }
        
        with open(os.path.join(self.processed_path, 'class_info.json'), 'w') as f:
            json.dump(class_info, f, indent=2)
        
        # Simpan label encoder
        with open(os.path.join(self.processed_path, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        return np.array(images), np.array(labels_encoded), class_info
    
    def get_dataset_info(self):
        """
        Dapatkan informasi tentang dataset tanpa load semua gambar
        """
        class_counts = {}
        total_images = 0
        
        for class_folder in os.listdir(self.dataset_path):
            class_path = os.path.join(self.dataset_path, class_folder)
            
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                class_counts[class_folder] = count
                total_images += count
        
        return {
            'total_images': total_images,
            'num_classes': len(class_counts),
            'class_distribution': class_counts
        }
    
    def split_dataset(self, X, y, test_size=0.2, val_size=0.1, random_state=42):
        """
        Split dataset menjadi train, validation, dan test
        """
        # Split train dan temp (test + val)
        X_train, X_temp, y_train, y_temp = train_test_split(
            X, y, test_size=(test_size + val_size), 
            random_state=random_state, stratify=y
        )
        
        # Split temp menjadi test dan val
        X_test, X_val, y_test, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=(test_size / (test_size + val_size)), 
            random_state=random_state, stratify=y_temp
        )
        
        print(f"Dataset split:")
        print(f"  Train: {len(X_train)} samples")
        print(f"  Validation: {len(X_val)} samples") 
        print(f"  Test: {len(X_test)} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test