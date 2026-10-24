# Traffic-Vision — Traffic Light Image ClassificatioN
MACHINE LEARNING | IMAGE CLASSIFICATION | PYTHON・SCIKIT-LEARN

Traffic-Vision is a traditional machine learning project that classifies traffic light images into three categories: Stop (Red), Ready (Yellow), and Go (Green). The project compares three classification algorithms with and without image preprocessing to investigate how image enhancement and color segmentation affect classification performance.

## 01. Project Overview

### Description

Traffic-Vision explores traffic light image classification using color-based feature extraction and traditional machine learning algorithms. It compares model performance under two experimental conditions: without preprocessing and with preprocessing.

### Objectives

- Classify traffic light images into Stop, Ready, and Go categories.
- Apply image processing techniques to enhance relevant color information.
- Extract color-based features using average BGR values and color histograms.
- Compare K-Nearest Neighbors (KNN), Random Forest, and Support Vector Machine (SVM).
- Evaluate model performance using accuracy, precision, recall, F1-score, and confusion matrices.

### Problem Statement

Traffic light images may contain variations in brightness, contrast, saturation, and background colors. These variations can affect color-based classification. This project investigates whether image preprocessing can improve classification performance compared with using images without additional preprocessing.

### Dataset

The project uses a dataset of 1,200 images across three classes.

| Class | Label | Images |
|---|---|---:|
| Stop | Red | 400 |
| Ready | Yellow | 400 |
| Go | Green | 400 |
| **Total** | | **1,200** |

The images were consolidated from the original training and testing datasets, and exact duplicate images were removed before splitting the dataset into training and testing sets.

- **Training set:** 960 images (80%)
- **Testing set:** 240 images (20%)
- **Testing distribution:** 80 images per class

### Methodology

Two experiments were conducted using the same train-test split, feature extraction method, classification algorithms, and evaluation metrics.

**Experiment 1 — Without Preprocessing**

Images are resized to 100 × 200 pixels for consistent feature extraction. No additional image enhancement or color segmentation is applied.

**Experiment 2 — With Preprocessing**

The preprocessing pipeline consists of:

1. **Resizing:** Resize images to 100 × 200 pixels.
2. **Noise Removal:** Apply Gaussian Blur using a 5 × 5 kernel.
3. **Contrast Enhancement:** Apply CLAHE to the L channel in LAB color space.
4. **Saturation Enhancement:** Increase HSV saturation by a factor of 1.5.
5. **Intensity Normalization:** Normalize the L channel in LAB color space.
6. **Color Segmentation:** Use HSV masks to isolate red, yellow, and green regions.

### Feature Extraction

Both experiments use the same color-based feature extraction method:

- Average BGR values: 3 features.
- Blue color histogram: 256 bins.
- Green color histogram: 256 bins.
- Red color histogram: 256 bins.

Each image produces a feature vector containing **771 features**. Histogram values are normalized by the total number of pixels, and `StandardScaler` is fitted on training features before transforming the testing features.

### Classification Algorithms

- **K-Nearest Neighbors (KNN):** `n_neighbors=3`
- **Random Forest:** `n_estimators=100`
- **Support Vector Machine (SVM):** Linear kernel

### Dataset Limitations and Result Reliability

**The reported results depend heavily on the dataset used in this experiment.** Although exact duplicate images were removed, similar images or modified versions originating from the same source may still remain in the dataset.

If related images appear in both the training and testing sets, the models may perform well on the testing set without generalizing equally well to genuinely unseen images. This can lead to overly optimistic evaluation results, including very high accuracy.

Therefore, the current dataset should be considered an **experimental testing resource for comparing classification methods**, rather than definitive evidence of real-world performance. The reported metrics describe performance on this particular dataset and should not be interpreted as guaranteed accuracy under different lighting conditions, camera settings, backgrounds, or real-world traffic environments.

Further validation should use a carefully verified dataset with source-level duplicate checks and, where possible, independently collected images that are completely separate from the training data.

## 02. Tech Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Image Processing | OpenCV |
| Numerical Computing | NumPy |
| Data Analysis | Pandas |
| Data Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Development Environment | Jupyter Notebook / Python environment |

## 03. Screenshots / Results

### Dataset Exploration

The dataset is examined before splitting to inspect class distribution and sample images.

<img width="790" height="490" alt="image" src="https://github.com/user-attachments/assets/a7a133f3-d57e-4678-8fca-69ea718eac45" />


<img width="844" height="886" alt="image" src="https://github.com/user-attachments/assets/6504b3a2-494e-4709-921b-d7e07afb0c60" />


### Image Preprocessing

The preprocessing experiment applies image enhancement and color segmentation before feature extraction.

<img width="963" height="495" alt="image" src="https://github.com/user-attachments/assets/ab3e2dab-f1ab-4180-a73c-f29b6de2364c" />

<img width="963" height="495" alt="image" src="https://github.com/user-attachments/assets/611b8459-1590-4b75-8f41-5026653e3315" />

<img width="963" height="495" alt="image" src="https://github.com/user-attachments/assets/e2f8a3d3-84cd-4091-ad73-15a2bd32b234" />


### Model Evaluation

Both experiments use the same testing set of 240 images, consisting of 80 images per class.

#### Experiment 1 — Without Preprocessing

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
|---|---:|---:|---:|---:|
| KNN | 88.33% | 88.43% | 88.33% | 88.37% |
| Random Forest | 97.92% | 97.98% | 97.92% | 97.90% |
| SVM | 97.92% | 97.98% | 97.92% | 97.90% |

#### Experiment 2 — With Preprocessing

| Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score |
|---|---:|---:|---:|---:|
| KNN | 93.33% | 93.44% | 93.33% | 93.36% |
| Random Forest | 99.58% | 99.59% | 99.58% | 99.58% |
| SVM | 98.33% | 98.37% | 98.33% | 98.33% |

*Note: The metrics above are the observed results on the current test split. Potential residual similarity or source overlap in the dataset may inflate the reported performance.*

### Confusion Matrices

Confusion matrices illustrate the distribution of correct and incorrect predictions for each class.

**Without Preprocessing**

<img width="1755" height="490" alt="image" src="https://github.com/user-attachments/assets/2395db5b-7f7f-45e0-8974-d9127e19fa07" />


**With Preprocessing**

<img width="1755" height="490" alt="image" src="https://github.com/user-attachments/assets/a3ffadbd-f9c0-4a1d-9db7-d71283068450" />


### Insights

- **Random Forest achieved the highest observed accuracy** in both experiments: 97.92% without preprocessing and 99.58% with preprocessing.
- **KNN showed the largest improvement**, increasing from 88.33% to 93.33%, a gain of 5.00 percentage points.
- **Random Forest improved by approximately 1.66 percentage points**, while SVM improved by approximately 0.41 percentage points.
- The results suggest that preprocessing may help the classifiers identify useful color information in this dataset.

### Usefulness

The experiments provide a comparative view of how image preprocessing and different traditional machine learning algorithms affect traffic light image classification. The results can help guide further experiments involving feature engineering, preprocessing methods, and model selection.

### Practical Application

The approach can serve as a baseline for educational computer vision projects and preliminary traffic light recognition research. However, it is not a validated traffic safety system and should not be used for real-world traffic control without substantially more rigorous testing.

## 04. Installation & Usage

### Prerequisites

- Python 3.9 or later
- pip
- Jupyter Notebook or another Python development environment

### Project Structure

```text

├── dataset/
│   ├── stop/
│   ├── ready/
│   └── go/
├── screenshots/
│   ├── dataset-distribution.png
│   ├── original-image-samples.png
│   ├── image-preprocessing.png
│   ├── confusion-matrix-without-preprocessing.png
│   └── confusion-matrix-with-preprocessing.png
├── notebooks/
│   ├── traffic_vision_without_preprocessing.ipynb
│   └── traffic_vision_with_preprocessing.ipynb
└── README.md
```

*Adjust the filenames and folder structure to match the files actually included in the repository.*

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YOUR_USERNAME/ml-TrafficLightImageClassification.git
   cd ml-TrafficLightImageClassification
   ```

2. Install the dependencies:

   ```bash
   pip install opencv-python numpy pandas matplotlib seaborn scikit-learn
   ```

### Dataset Configuration

Place the cleaned images in the following directories:

- `dataset/stop/`
- `dataset/ready/`
- `dataset/go/`

Supported image formats are JPG, JPEG, and PNG.

### Running the Experiments

1. Open the project notebook in Jupyter Notebook or your preferred Python environment.
2. Configure the dataset directory and class labels.
3. Load the images and inspect the dataset distribution.
4. Split the dataset into training and testing sets using stratified sampling.
5. Run the experiment without preprocessing.
6. Run the experiment with preprocessing.
7. Compare accuracy, precision, recall, F1-score, and confusion matrices.

For a fair comparison, both experiments should use the same training and testing images.

### Reproducibility and Evaluation Notes

The experiments use `random_state=42` for dataset splitting and Random Forest configuration. Before drawing conclusions from the metrics, verify that duplicate and near-duplicate images, including transformed versions from the same source, do not cross the training-testing boundary.

The results should be treated as preliminary experimental findings until they are validated on a more rigorously separated, independently collected dataset.

---
