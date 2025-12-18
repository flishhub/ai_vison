# 🚀 AI Vision Lab: From Scratch to Professional Inference

This repository documents my journey in building a local Computer Vision environment using **PyTorch**, **CUDA (NVIDIA GPU Acceleration)**, and **Conda**. The project evolved through four distinct iterations, moving from basic neural networks to professional-grade transfer learning.

## 🛠️ Tech Stack

* **Framework:** PyTorch
* **Hardware:** NVIDIA GPU (CUDA-accelerated)
* **Environment:** Conda / PyCharm
* **Datasets:** CIFAR-10 (60,000 32x32 images)

---

## 📈 The Evolution of the Model

### **Version 1: The "Hello World" (Initial Setup)**

* **Architecture:** Simple 2-layer Convolutional Neural Network (CNN).
* **Goal:** Verify CUDA installation and establish a baseline.
* **Accuracy:** ~54%
* **Key Learning:** Established the "Process Guard" (`if __name__ == '__main__':`) required for Windows multi-processing.

### **Version 2: The "Optimizer" (Hyperparameter Tuning)**

* **Architecture:** Same 2-layer CNN.
* **Change:** Increased training duration to **20 Epochs**.
* **Accuracy:** **64%**
* **Key Learning:** Hit the "Architecture Ceiling." Even with more training time, the simple model couldn't capture the complexity of the data.

### **Version 3: The "Advanced Architecture" (Performance Leap)**

* **Architecture:** 3-layer CNN with **Batch Normalization** and **Dropout**.
* **Optimization:** Switched to **Adam Optimizer** and added **Data Augmentation** (random flips/rotations).
* **Accuracy:** **81%** 🚀
* **Key Learning:** Outperformed the original AlexNet benchmarks by focusing on generalization rather than just memorization.

### **Version 4: The "Pro" Tier (Transfer Learning)**

* **Architecture:** **ResNet-50** (50-layer deep Residual Network).
* **Method:** Transfer Learning using ImageNet weights.
* **Accuracy:** High-precision breed identification.
* **The "Rufus" Test:** Successfully identified my pet as a **Blenheim Spaniel** with high confidence, whereas previous versions struggled with the low-resolution 32x32 constraint.

---

## 🧪 Experiments & Insights

### **The "Rufus" Case Study**

One of the main goals was "Inference" (testing the AI on real-world photos).

1. **V3 Model (Custom):** Identified my dog as a "Deer" (31% confidence) and later a "Cat" (51% confidence). This was due to the model being limited to **32x32 pixel** inputs.
2. **ResNet-50 (Pro):** Corrected the classification to a **Blenheim Spaniel**.
**Lesson Learned:** High-resolution input (224 \times 224) and deeper architectures are required for professional-grade species and breed identification.

---

## 📂 Project Structure

* `ai_vision_v1.py`: Baseline training script.
* `ai_vision_v3.py`: Optimized 81% accuracy model.
* `pet_inference.py`: Custom script to test images of my pets.
* `pro_inference.py`: Professional ResNet-50 deployment script.

## 🏁 How to Run

1. Ensure you have the `ai_vision` conda environment active.
2. Run training: `python ai_vision_v3.py`
3. Run inference: `python pro_inference.py`

---

### **Final Thoughts**

This experiment proved that while building a custom model is the best way to learn the "math" of AI, **Transfer Learning** is the standard for real-world applications where accuracy and breed specificity are required.
