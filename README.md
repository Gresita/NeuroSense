# NeuroSense

NeuroSense is a machine learning and deep learning project for emotion recognition using EEG and Eye-Tracking data from the SEED-IV dataset.

The project explores different approaches for emotion classification and clustering, including traditional machine learning algorithms, neural networks, and deep learning architectures.

---

## Prerequisites

Before running the project, make sure you have installed:

- Python 3.10 or higher
- Jupyter Notebook
- Git

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd NeuroSense
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux / macOS

```bash
source .venv/bin/activate
```

### 4. Install required packages

```bash
pip install -r requirements.txt
```


---

## Running the Project

The notebooks should be executed in the following order:

### 1. Data Preprocessing

Run:

```text
preprocessing.ipynb
```

This notebook performs:

- Data cleaning
- Missing value analysis
- Outlier analysis
- Data scaling and normalization

---

### 2. Feature Engineering

Run:

```text
feature_engineering.ipynb
```

This notebook generates additional statistical features from EEG and Eye-Tracking signals.

---

### 3. Data Visualization

Run:

```text
visualization.ipynb
```

This notebook contains exploratory data analysis and visualizations used throughout the project.

---

### 4. Classification Models

Navigate to the classification folder and execute the notebooks individually.

Implemented models:

- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machine (SVM)
- Random Forest
- XGBoost
- Multi-Layer Perceptron (MLP)
- CNN-BiLSTM

---

### 5. Clustering

Run:

```text
clustering.ipynb
```

Implemented clustering algorithms:

- K-Means
- DBSCAN
- Agglomerative Clustering

---

### 6. Final Model Comparison

Run:

```text
classification/final_model_comparison.ipynb
```

This notebook generates comparative visualizations and performance rankings for all implemented classification models.

---

## Running the Dashboard

The project includes an interactive dashboard built with Streamlit.

From the root project directory run:

```bash
streamlit run dashboard/Live_Dashboard.py
```

After the application starts, open the URL displayed in the terminal (typically):

```text
http://localhost:8501
```

The dashboard provides interactive visualizations and analysis of the models, datasets, and experimental results generated throughout the project.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- PyTorch
- Streamlit
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Authors

NeuroSense Project

Faculty of Computer Science and Engineering  
UBT – University for Business and Technology