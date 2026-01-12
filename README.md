# Eye Disease Detection & Prediction Using Machine Learning

## 📄 Abstract
This project implements a deep learning framework for the early detection of eye diseases (Diabetic Retinopathy, Glaucoma, and Cataract) using Fundus and OCT images. [cite_start]The system utilizes **Convolutional Neural Networks (CNNs)** and **Transfer Learning** (EfficientNetB3, ResNet50, VGG16) to achieve high diagnostic precision[cite: 6, 7]. [cite_start]A key focus of this project is **Fairness-Aware Machine Learning**, incorporating techniques to reduce bias across different patient demographics[cite: 8].

## 🚀 Key Features
* **Multi-Disease Classification:** Detects Diabetic Retinopathy, Glaucoma, and Cataract.
* [cite_start]**Transfer Learning:** Fine-tuned implementations of VGG16, ResNet50, and EfficientNetB3[cite: 172].
* [cite_start]**Fairness Optimization:** Mitigates demographic bias (Age, Gender) using reweighting and threshold calibration[cite: 66].
* [cite_start]**Explainable AI:** Includes Grad-CAM visualizations to show *where* the model is looking (e.g., optic disc, macula)[cite: 231].

## 📊 Performance Results
The models were evaluated on a balanced dataset of approx. [cite_start]2000 images per condition[cite: 95].

| Model | Accuracy | Sensitivity | Specificity | AUC |
| :--- | :--- | :--- | :--- | :--- |
| **EfficientNetB3 (Transfer)** | **86%** | **0.85** | **0.87** | **0.91** |
| ResNet50 (Transfer) | 85% | 0.83 | 0.86 | 0.89 |
| VGG16 (Transfer) | 82% | 0.80 | 0.84 | 0.87 |
| Baseline CNN | 75% | 0.68 | 0.78 | 0.80 |
| Fairness-Aware Model | 80% | 0.79 | 0.81 | 0.85 |
[cite_start]*(Source: Project Report Table I [cite: 192])*

## 🛠️ Methodology
1.  [cite_start]**Preprocessing:** CLAHE contrast enhancement, normalization, and geometric augmentation (rotation, zoom, flip) [cite: 157-160].
2.  **Architecture:**
    * [cite_start]**Custom CNN:** 4 Convolutional blocks with increasing filters (32-256) and Batch Normalization[cite: 130].
    * [cite_start]**Transfer Learning:** Feature extraction using ImageNet weights, fine-tuned on ocular data[cite: 49].
3.  [cite_start]**Fairness:** Adversarial debiasing and group-aware sampling to ensure equitable performance across age and gender groups[cite: 112].

## 🖼️ Visuals
### Model Architecture
![Architecture](assets/architecture.png)
[cite_start]*Figure 1: Custom CNN Architecture illustrating hierarchical feature extraction[cite: 219].*

### Grad-CAM Output
![Grad-CAM](assets/output_sample.png)
[cite_start]*Figure 5: Grad-CAM visualization highlighting the optic disc and macula as key regions for diagnosis[cite: 231].*

## 📚 References
[cite_start]Based on the project report "Eye Diseases Detection and Prediction Using Machine Learning" by S. Velani, N. Sutaria, and A. Satra[cite: 1].
