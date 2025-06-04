"""
Flask Web Application untuk Plant Disease Classification
Updated untuk menggunakan model hasil training dan preprocessing yang benar
"""

from flask import Flask, request, render_template, jsonify, redirect, url_for
from flask_cors import CORS
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from PIL import Image
import io
import os
import json
import pickle
import base64
from datetime import datetime
from image_preprocessor import ImagePreprocessor

app = Flask(__name__)
CORS(app)

# Konfigurasi
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'model/leaf_model.h5'
PROCESSED_DATA_PATH = 'processed_data'

# Pastikan folder upload ada
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class PlantDiseasePredictor:
    def __init__(self):
        self.model = None
        self.class_info = None
        self.preprocessor = ImagePreprocessor(target_size=(256, 256))
        self.model_loaded = False
        
    def load_model_and_info(self):
        """
        Load trained model dan informasi kelas
        """
        try:
            # Load model
            if os.path.exists(MODEL_PATH):
                self.model = keras.models.load_model(MODEL_PATH)
                print(f"✅ Model loaded: {MODEL_PATH}")
            else:
                print(f"❌ Model file tidak ditemukan: {MODEL_PATH}")
                return False
            
            # Load class info
            class_info_path = os.path.join(PROCESSED_DATA_PATH, 'class_info.json')
            if os.path.exists(class_info_path):
                with open(class_info_path, 'r') as f:
                    self.class_info = json.load(f)
                print(f"✅ Class info loaded: {len(self.class_info['class_names'])} classes")
            else:
                print(f"❌ Class info tidak ditemukan: {class_info_path}")
                return False
            
            # Load model info jika ada
            model_info_path = os.path.join('model', 'model_info.json')
            if os.path.exists(model_info_path):
                with open(model_info_path, 'r') as f:
                    self.model_info = json.load(f)
                print(f"✅ Model info loaded")
            else:
                print("⚠️ Model info tidak ditemukan, menggunakan default")
                self.model_info = {"training_samples": "Unknown", "test_accuracy": "Unknown"}
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def preprocess_image(self, image):
        """
        Preprocess gambar sesuai dengan training pipeline
        """
        try:
            # Convert PIL ke numpy array
            if hasattr(image, 'mode') and image.mode == 'RGBA':
                image = image.convert('RGB')
            
            img_array = np.array(image)
            
            # Apply preprocessing yang sama dengan training
            processed = self.preprocessor.preprocess_single_image(
                img_array, apply_blur=True, apply_hist_eq=True
            )
            
            # Add batch dimension
            processed = np.expand_dims(processed, axis=0)
            
            return processed
            
        except Exception as e:
            print(f"❌ Error preprocessing image: {str(e)}")
            return None
    
    def predict(self, image):
        """
        Prediksi penyakit tanaman dari gambar
        """
        if not self.model_loaded:
            return None, "Model belum dimuat"
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image)
            if processed_image is None:
                return None, "Gagal memproses gambar"
            
            # Prediksi
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top predictions
            top_indices = np.argsort(predictions[0])[::-1][:5]  # Top 5
            results = []
            
            for i, idx in enumerate(top_indices):
                class_name = self.class_info['label_mapping'][str(idx)]
                confidence = float(predictions[0][idx])
                
                # Clean class name
                if '___' in class_name:
                    plant, disease = class_name.split('___', 1)
                    clean_name = f"{plant.replace('_', ' ')} - {disease.replace('_', ' ')}"
                else:
                    clean_name = class_name.replace('_', ' ')
                
                results.append({
                    'rank': i + 1,
                    'class_name': clean_name,
                    'raw_class_name': class_name,
                    'confidence': confidence,
                    'percentage': confidence * 100
                })
            
            return results, None
            
        except Exception as e:
            error_msg = f"Error dalam prediksi: {str(e)}"
            print(f"❌ {error_msg}")
            return None, error_msg

# Inisialisasi predictor
predictor = PlantDiseasePredictor()

@app.route('/')
def index():
    """
    Halaman utama
    """
    # Load model info untuk ditampilkan
    model_status = predictor.load_model_and_info()
    
    model_info = {
        'loaded': model_status,
        'total_classes': predictor.class_info['num_classes'] if predictor.class_info else 0,
        'model_path': MODEL_PATH,
        'training_samples': predictor.model_info.get('training_samples', 'Unknown') if hasattr(predictor, 'model_info') else 'Unknown',
        'test_accuracy': predictor.model_info.get('test_accuracy', 'Unknown') if hasattr(predictor, 'model_info') else 'Unknown'
    }
    
    return render_template('index.html', model_info=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint untuk prediksi
    """
    try:
        # Cek apakah ada file yang diupload
        if 'file' not in request.files:
            return jsonify({'error': 'Tidak ada file yang diupload'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'Tidak ada file yang dipilih'}), 400
        
        # Validasi tipe file
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
        if not ('.' in file.filename and 
                file.filename.rsplit('.', 1)[1].lower() in allowed_extensions):
            return jsonify({'error': 'Tipe file tidak didukung. Gunakan: PNG, JPG, JPEG, GIF, BMP'}), 400
        
        # Load dan process image
        try:
            image = Image.open(io.BytesIO(file.read()))
            original_size = image.size
        except Exception as e:
            return jsonify({'error': f'Gagal membaca gambar: {str(e)}'}), 400
        
        # Save uploaded image
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"upload_{timestamp}_{file.filename}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        image.save(filepath)
        
        # Prediksi
        predictions, error = predictor.predict(image)
        
        if error:
            return jsonify({'error': error}), 500
        
        # Response
        response = {
            'success': True,
            'filename': filename,
            'filepath': filepath,
            'original_size': original_size,
            'predictions': predictions,
            'top_prediction': predictions[0] if predictions else None,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/model-info')
def model_info():
    """
    API untuk informasi model
    """
    if not predictor.model_loaded:
        predictor.load_model_and_info()
    
    if predictor.model_loaded:
        info = {
            'model_loaded': True,
            'num_classes': predictor.class_info['num_classes'],
            'class_names': predictor.class_info['class_names'][:10],  # Sample 10 kelas
            'total_classes': len(predictor.class_info['class_names']),
            'model_architecture': predictor.model_info.get('model_architecture', 'CNN'),
            'training_samples': predictor.model_info.get('training_samples', 'Unknown'),
            'test_accuracy': predictor.model_info.get('test_accuracy', 'Unknown'),
            'preprocessing_steps': predictor.model_info.get('preprocessing_steps', [])
        }
    else:
        info = {
            'model_loaded': False,
            'error': 'Model tidak dapat dimuat'
        }
    
    return jsonify(info)

@app.route('/health')
def health_check():
    """
    Health check endpoint
    """
    status = {
        'status': 'healthy',
        'model_loaded': predictor.model_loaded,
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(status)

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('index.html', error="Halaman tidak ditemukan"), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'error': 'File terlalu besar. Maksimal 16MB'}), 413

if __name__ == '__main__':
    print("="*50)
    print("🌱 PLANT DISEASE CLASSIFIER WEB APP")
    print("="*50)
    
    # Load model saat startup
    print("📥 Loading model...")
    if predictor.load_model_and_info():
        print("✅ Model berhasil dimuat")
        print(f"📊 Total kelas: {predictor.class_info['num_classes']}")
    else:
        print("❌ Gagal memuat model")
        print("💡 Jalankan training terlebih dahulu: python train_model.py")
    
    print("="*50)
    print("🚀 Starting Flask server...")
    print("🌐 Open: http://localhost:5000")
    print("="*50)
    
    # Set max file size (16MB)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
    
    app.run(debug=True, host='0.0.0.0', port=5000)