# 🌱 PlantVision-CNN: Optimized Plant Disease Classification

> **RTX 3050 Ti optimized plant disease classification system using Convolutional Neural Networks with memory-efficient preprocessing pipeline.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.8+-orange.svg)](https://tensorflow.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-yellow.svg)]()

## 🎯 **Overview**

PlantVision-CNN is a memory-optimized plant disease classification system specifically designed for RTX 3050 Ti (4GB VRAM) constraints. The system combines smart preprocessing techniques with ultra-lightweight CNN architecture to achieve effective plant disease detection while maintaining performance on limited hardware.

### **Key Features**
- **🚀 RTX 3050 Ti Optimized**: Ultra-small batch size (1) and memory-efficient architecture
- **⚡ Smart GPU/CPU Fallback**: Automatic fallback system for hardware compatibility  
- **🔧 4-Stage Preprocessing**: Resize → Histogram Equalization → Gaussian Blur → Normalization
- **🌐 Flask Web Interface**: Real-time classification with confidence scoring
- **📊 Small Dataset Support**: Built-in dataset reduction for faster development

## 🏗️ **System Architecture**

```
Input Image → Small Dataset → Preprocessing Pipeline → Lightweight CNN → Results
     ↓             ↓              ↓                    ↓              ↓
   Raw Image   500/class     [Resize, CLAHE,       Feature Maps    Disease Class
   (Various)   25 classes     Blur, Norm]         (16→32→64→128)   + Confidence
```

### **Memory-Optimized Pipeline**
1. **Dataset Reduction**: 500 samples per class, max 25 classes for development
2. **Smart Resize**: 256×256 for preprocessing, 64×64 for memory efficiency
3. **CLAHE Enhancement**: Contrast-limited adaptive histogram equalization
4. **Conservative Blur**: (3×3) Gaussian kernel for noise reduction
5. **Float32 Normalization**: Memory-efficient [0,1] range scaling

### **Ultra-Lightweight CNN**
- **Architecture**: 4 conv blocks (16→32→64→128 filters)
- **Memory Optimization**: GlobalAveragePooling2D instead of Flatten
- **Batch Size**: 1 (ultra-conservative for 4GB VRAM)
- **Parameters**: ~500K (vs 25M+ in standard models)
- **Dropout**: Conservative regularization (0.25→0.5)

## 📊 **Dataset & Performance**

**Small PlantVillage Dataset** (Development):
- **Source**: Reduced from full PlantVillage dataset
- **Images**: ~12,500 (500 per class × 25 classes)
- **Split**: 70% Train / 15% Validation / 15% Test
- **Format**: Preprocessed to 256×256, stored as float32

**Hardware Performance**:
- **RTX 3050 Ti**: ✅ Successfully trains with 1 batch size
- **Training Time**: 2-4 hours (depending on dataset size)
- **Memory Usage**: <3.5GB VRAM (safe for 4GB cards)
- **CPU Fallback**: Automatic if GPU fails

**Model Performance**:
- **Test Accuracy**: 65-85% (depending on dataset complexity)
- **Model Size**: ~4MB (lightweight for deployment)
- **Inference Speed**: <50ms per image

## 🚀 **Quick Start**

### **Prerequisites**
- Python 3.8+
- RTX 3050 Ti / RTX 3060 / GTX 1660+ (or CPU fallback)
- 8GB+ RAM

### **Installation**

```bash
# Clone repository
git clone https://github.com/yourusername/Project-PlantVision-CNN.git
cd Project-PlantVision-CNN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### **Dataset Setup**

1. **Download PlantVillage dataset** from [Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
2. **Extract to `dataset/raw/`** directory
3. **Create small dataset** for development:
   ```bash
   python create_small_dataset.py
   ```

### **Training Pipeline**

```bash
# Option 1: Full automated pipeline
python run_pipeline.py

# Option 2: Step-by-step execution
python create_small_dataset.py    # Create manageable dataset
python preprocessing.py           # 4-stage preprocessing
python train_model.py            # RTX 3050 Ti optimized training
python app.py                    # Launch web interface

# Option 3: Check processed data
python preprocessing.py --check   # Verify preprocessing
```

### **Web Application**

```bash
python app.py
```
Access at: `http://localhost:5000`

## 🔧 **RTX 3050 Ti Configuration**

### **Memory Optimization Settings**
```python
# Ultra-conservative for 4GB VRAM
BATCH_SIZE = 1                    # Minimum possible
TARGET_SIZE = (256, 256)          # Balanced quality/memory
MODEL_FILTERS = [16, 32, 64, 128] # Lightweight progression
MEMORY_GROWTH = True              # Dynamic allocation
MIXED_PRECISION = False           # Disabled (causes issues)
```

### **Training Parameters**
```python
# Conservative training settings
EPOCHS = 25                       # Reasonable duration
SAMPLES_PER_CLASS = 500          # Manageable dataset size
MAX_CLASSES = 25                 # Memory-friendly
LEARNING_RATE = 0.001            # Stable convergence
PATIENCE = 8                     # Early stopping
```

## 📈 **Preprocessing Techniques**

### **4-Stage Pipeline Implementation**

1. **Resize (256×256)**
   ```python
   cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
   ```

2. **CLAHE Histogram Equalization**
   ```python
   clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
   enhanced = clahe.apply(lightness_channel)
   ```

3. **Gaussian Blur (3×3)**
   ```python
   cv2.GaussianBlur(image, (3, 3), sigma=0)
   ```

4. **Normalization [0,1]**
   ```python
   normalized = image.astype(np.float32) / 255.0
   ```

### **Visualization**
Preprocessing visualization automatically saved to:
```
processed_data/preprocessing_visualization.png
```

## 🛠️ **Hardware Compatibility**

### **Tested Configurations**
| GPU Model | VRAM | Batch Size | Status | Training Time |
|-----------|------|------------|--------|---------------|
| RTX 3050 Ti | 4GB | 1 | ✅ Works | 3-4 hours |
| RTX 3060 | 12GB | 4-8 | ✅ Optimal | 1-2 hours |
| GTX 1660 | 6GB | 2-4 | ✅ Good | 2-3 hours |
| CPU Only | - | 8 | ✅ Fallback | 6-8 hours |

### **Troubleshooting Common Issues**

**GPU Memory Error:**
```bash
# Reduce dataset size
python create_small_dataset.py  # Already optimized for 4GB
```

**Training Fails:**
```bash
# Check preprocessed data
python preprocessing.py --check

# Manual CPU training
export CUDA_VISIBLE_DEVICES=""
python train_model.py
```

**Slow Performance:**
```bash
# Verify GPU usage
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## 📱 **Web Interface Features**

### **Upload & Predict**
- Drag & drop image upload
- Real-time preprocessing preview
- Confidence scores for top predictions
- Disease information display

### **Supported Formats**
- JPG, JPEG, PNG
- Max file size: 16MB
- Auto-resize to model requirements

### **API Endpoints**
```python
POST /predict              # Image classification
GET /model-info           # Model architecture info
GET /health               # System status
```

## 🔬 **Research Applications**

### **Academic Use Cases**
- **Computer Vision Course Projects**: Preprocessing technique comparison
- **Machine Learning Education**: Hardware constraint optimization
- **Agricultural Technology**: Rapid prototyping for plant disease detection
- **Edge Computing Research**: Lightweight model deployment

### **Future Research Directions**
- **Mobile Deployment**: TensorFlow Lite conversion for smartphones
- **Transfer Learning**: Fine-tuning pre-trained models with memory constraints
- **Edge Computing**: Raspberry Pi deployment optimization
- **Real-time Detection**: Video stream processing for continuous monitoring

## 📊 **Project Structure**

```
Project-PlantVision-CNN/
├── 📁 dataset/
│   ├── raw/                     # Original PlantVillage dataset
│   └── small/                   # Reduced dataset (auto-generated)
├── 📁 processed_data/           # Preprocessed numpy arrays
├── 📁 model/                    # Trained models and info
├── 📁 static/
│   ├── uploads/                 # Web app uploads
│   └── style.css               # Styling
├── 📁 templates/
│   └── index.html              # Web interface
├── 🐍 create_small_dataset.py   # Dataset reduction utility
├── 🐍 dataset_loader.py         # Data loading utilities  
├── 🐍 image_preprocessor.py     # 4-stage preprocessing
├── 🐍 preprocessing.py          # Main preprocessing script
├── 🐍 train_model.py           # RTX 3050 Ti optimized trainer
├── 🐍 app.py                   # Flask web application
├── 🐍 run_pipeline.py          # Automation pipeline
└── 📄 requirements.txt         # Dependencies
```

## 🧪 **Development Workflow**

### **For Research Development**
```bash
# 1. Create small dataset for rapid iteration
python create_small_dataset.py

# 2. Experiment with preprocessing
python preprocessing.py

# 3. Quick model training
python train_model.py

# 4. Test web interface
python app.py
```

### **For Production Deployment**
```bash
# 1. Use full dataset
# 2. Optimize model architecture
# 3. Implement proper logging
# 4. Add error handling
# 5. Deploy with proper web server
```

## 🤝 **Contributing**

We welcome contributions! Areas for improvement:
- **Model Architecture**: More efficient designs for 4GB VRAM
- **Preprocessing Optimization**: Faster batch processing
- **Web Interface**: Enhanced user experience
- **Documentation**: Code examples and tutorials

### **Development Setup**
```bash
# Install development dependencies
pip install -r requirements.txt

# Run tests (if available)
python -m pytest tests/

# Code formatting
black . && isort .
```

## 📚 **Citation**

If you use this work in your research, please cite:

```bibtex
@misc{plantvision2024,
  title={PlantVision-CNN: RTX 3050 Ti Optimized Plant Disease Classification},
  author={Muhammad Taufiq Al Fikri},
  year={2024},
  url={https://github.com/yourusername/Project-PlantVision-CNN},
  note={Telkom University - Digital Image Processing Course Project}
}
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **PlantVillage Dataset**: Hughes et al., Pennsylvania State University
- **TensorFlow Team**: Google Brain
- **OpenCV Community**: Computer vision library
- **Telkom University**: Digital Image Processing Course (CAK4OBB3)
- **RTX 3050 Ti Community**: Hardware optimization insights

## 📞 **Contact**

- **Author**: Muhammad Taufiq Al Fikri
- **Email**: taufikaifikri28@gmail.com  
- **LinkedIn**: [https://www.linkedin.com/in/taufiq-afk/](https://www.linkedin.com/in/taufiq-afk/)
- **Research Gate**: Coming Soon

---

**⭐ Star this repository if you find it helpful for your RTX 3050 Ti projects!**
