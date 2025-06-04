import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from tqdm import tqdm

class ImagePreprocessor:
    def __init__(self, target_size=(64, 64)):
        self.target_size = target_size
        self.scaler = MinMaxScaler()
        
    def resize_image(self, image, target_size=None):
        """
        Resize gambar ke ukuran target
        """
        if target_size is None:
            target_size = self.target_size
            
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    def histogram_equalization(self, image):
        """
        Histogram Equalization untuk meningkatkan kontras
        """
        # Convert ke LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Pisahkan channel L (Lightness)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Gabungkan kembali channel
        lab = cv2.merge([l, a, b])
        
        # Convert kembali ke RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def gaussian_blur(self, image, kernel_size=(3, 3), sigma=0):
        """
        Gaussian Blur untuk noise reduction
        """
        return cv2.GaussianBlur(image, kernel_size, sigma)
    
    def normalize_image(self, image):
        """
        Normalisasi pixel values ke range [0, 1]
        """
        return image.astype(np.float32) / 255.0
    
    def preprocess_single_image(self, image, apply_blur=True, apply_hist_eq=True):
        """
        Apply semua preprocessing steps pada satu gambar
        """
        # 1. Resize
        processed = self.resize_image(image)
        
        # 2. Histogram Equalization (optional)
        if apply_hist_eq:
            processed = self.histogram_equalization(processed)
        
        # 3. Gaussian Blur (optional)
        if apply_blur:
            processed = self.gaussian_blur(processed)
        
        # 4. Normalization
        processed = self.normalize_image(processed)
        
        return processed
    
    def preprocess_batch(self, images, apply_blur=True, apply_hist_eq=True, show_progress=True):
        """
        Preprocess batch of images
        """
        processed_images = []
        
        iterator = tqdm(images, desc="Preprocessing images") if show_progress else images
        
        for image in iterator:
            processed = self.preprocess_single_image(
                image, apply_blur=apply_blur, apply_hist_eq=apply_hist_eq
            )
            processed_images.append(processed)
        
        return np.array(processed_images)
    
    def save_preprocessed_data(self, X_train, X_val, X_test, y_train, y_val, y_test, save_path="processed_data"):
        """
        Simpan data yang sudah dipreprocess
        """
        os.makedirs(save_path, exist_ok=True)
        
        print("Saving preprocessed data...")
        
        # Simpan dalam format numpy
        np.save(os.path.join(save_path, 'X_train.npy'), X_train)
        np.save(os.path.join(save_path, 'X_val.npy'), X_val)
        np.save(os.path.join(save_path, 'X_test.npy'), X_test)
        np.save(os.path.join(save_path, 'y_train.npy'), y_train)
        np.save(os.path.join(save_path, 'y_val.npy'), y_val)
        np.save(os.path.join(save_path, 'y_test.npy'), y_test)
        
        # Simpan preprocessing parameters
        preprocessing_params = {
            'target_size': self.target_size,
            'data_shape': {
                'train': X_train.shape,
                'val': X_val.shape,
                'test': X_test.shape
            }
        }
        
        with open(os.path.join(save_path, 'preprocessing_params.pkl'), 'wb') as f:
            pickle.dump(preprocessing_params, f)
        
        print(f"Data saved to {save_path}")
        print(f"Train shape: {X_train.shape}")
        print(f"Val shape: {X_val.shape}")
        print(f"Test shape: {X_test.shape}")
    
    def load_preprocessed_data(self, save_path="processed_data"):
        """
        Load data yang sudah dipreprocess
        """
        X_train = np.load(os.path.join(save_path, 'X_train.npy'))
        X_val = np.load(os.path.join(save_path, 'X_val.npy'))
        X_test = np.load(os.path.join(save_path, 'X_test.npy'))
        y_train = np.load(os.path.join(save_path, 'y_train.npy'))
        y_val = np.load(os.path.join(save_path, 'y_val.npy'))
        y_test = np.load(os.path.join(save_path, 'y_test.npy'))
        
        with open(os.path.join(save_path, 'preprocessing_params.pkl'), 'rb') as f:
            params = pickle.load(f)
        
        return X_train, X_val, X_test, y_train, y_val, y_test, params
    
    def visualize_preprocessing_steps(self, image):
        """
        Visualisasi setiap step preprocessing untuk debugging
        """
        import matplotlib.pyplot as plt
        
        # Original
        original = image.copy()
        
        # Step 1: Resize
        resized = self.resize_image(image)
        
        # Step 2: Histogram Equalization
        hist_eq = self.histogram_equalization(resized)
        
        # Step 3: Gaussian Blur
        blurred = self.gaussian_blur(hist_eq)
        
        # Step 4: Normalization
        normalized = self.normalize_image(blurred)
        
        # Plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        axes[0, 0].imshow(original)
        axes[0, 0].set_title('Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(resized)
        axes[0, 1].set_title('Resized')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(hist_eq)
        axes[0, 2].set_title('Histogram Equalization')
        axes[0, 2].axis('off')
        
        axes[1, 0].imshow(blurred)
        axes[1, 0].set_title('Gaussian Blur')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(normalized)
        axes[1, 1].set_title('Normalized')
        axes[1, 1].axis('off')
        
        # Hide empty subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        return fig