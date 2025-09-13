# Chest X-ray Disease Classification (Multi-label)

This project focuses on **multi-label classification of chest X-ray images** using a combination of **OpenCV image processing**, **pre-trained deep learning models**, and **hybrid feature integration**. The pipeline leverages both traditional image processing techniques and modern deep learning methods to improve classification accuracy and robustness.

---

## 📝 Project Objective

The objective of this assignment is to create a pipeline that combines advanced **OpenCV preprocessing** with **deep learning models** to classify chest X-ray images into multiple disease categories. By integrating feature extraction from OpenCV with CNN features, we aim to enhance multi-label classification performance and provide interpretable results.

---

## 📂 Dataset

**Dataset Name:** Chest X-ray Dataset  
**Number of Images:** 3,710  
**Source:** [Kaggle Dataset](https://www.kaggle.com/datasets/rishabhrp/chest-x-ray-dataset)  

The dataset contains chest X-ray images with comprehensive metadata, suitable for **multi-label disease classification** tasks.

---

## ⚙️ Project Pipeline

### **Step 1: Data Preprocessing and Augmentation**
- Resize all images to **224x224 pixels**  
- Apply **Gaussian noise** to simulate real-world imaging conditions  
- Perform advanced augmentations:
  - Rotation  
  - Perspective transformation  
  - Histogram equalization  
- Split dataset into **training, validation, and test sets**  

### **Step 2: Feature Extraction (OpenCV)**
- Extract **ORB** or **SIFT** keypoints and descriptors  
- Save descriptors for visualization and analysis  

### **Step 3: Deep Learning Models**
We trained **three models**:

1. **DenseNet121** (pre-trained) – without augmentation  
2. **DenseNet121** – with augmentation, last 50 layers trainable  
3. **EfficientNetB0** – with augmentation and combined features:
   - OpenCV feature extraction (`extract_orb`)  
   - CNN feature extractor (EfficientNetB0)  
   - Final model uses **XGBoost** on combined features  

### **Step 4: Hybrid Model**
- Combine **OpenCV features** with **CNN features**  
- Train **XGBoost** for improved multi-label classification  

### **Step 5: Evaluation**
- Metrics: **Accuracy, Precision, Recall, F1-score**  
- Visualizations: **Confusion matrices, ROC curves**  
- Comparison between:
  - CNN-only models  
  - Hybrid OpenCV + CNN model  

---

## 📈 Results & Visualizations

All graphs and curves are saved in the `/curves/` folder:  
- **Training curves** for all models  
- **ROC curves** for hybrid and base models  

---

## 🛠️ Technologies & Libraries

- **Python 3.x**  
- **OpenCV** – preprocessing, feature extraction  
- **TensorFlow / Keras** – pre-trained CNN models  
- **EfficientNetB0 & DenseNet121** – transfer learning  
- **XGBoost** – hybrid feature classification  
- **NumPy, Pandas** – data manipulation  
- **Matplotlib, Seaborn** – visualization  
- **scikit-learn** – metrics, ROC, evaluation  

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/Chest-X-ray-Disease-Classification.git
cd Chest-X-ray-Disease-Classification
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the Jupyter notebook for preprocessing, training, and evaluation:
```bash
jupyter notebook
```
4. Explore results in the `/curves/` folder.

## 📌 Future Improvements

- Implement Grad-CAM to visualize CNN attention
- Automate the entire pipeline for batch processing
- Explore additional data augmentation techniques
- Optimize hybrid model with hyperparameter tuning

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgements

- Kaggle chest X-ray dataset
- Open-source libraries for computer vision and deep learning
- Tutorials and community contributions on medical image classification

```pgsql
If you want, I can also **add a clean GitHub-ready version with badges, dataset link badge, Python version badge, and quick demo images** to make it **look very professional on GitHub**.  

Do you want me to do that?
```
