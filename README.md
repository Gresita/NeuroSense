# NeuroSense: Emotion Recognition using EEG and Eye Tracking Data

## Project Overview

NeuroSense is a machine learning project focused on emotion recognition using multimodal physiological signals. The system analyzes EEG (Electroencephalography) and Eye Tracking features to classify emotional states.

The project was developed as part of the Machine Learning Models course and includes data preprocessing, feature engineering, classification, clustering, evaluation, and an interactive dashboard for emotion analysis.

---

## Dataset

Dataset: SEED-IV

Features:

* 310 EEG features
* 31 Eye Tracking features
* 14 Engineered features

Total features: 355

Classes:

* Neutral
* Negative
* Positive

Dataset size:

* 37,575 samples

---

## Project Structure

data/

* neurosense_cleaned.csv

classification/

* KNN
* Decision Tree
* Logistic Regression
* MLP Neural Network
* Random Forest
* Extra Trees

clustering/

* K-Means
* DBSCAN
* Agglomerative Clustering

dashboard/

* Live_Dashboard.py
* pages/Emotion_Analyzer.py
* models/random_forest_3class.pkl

results/

* evaluation metrics
* confusion matrices
* visualizations

---

## Machine Learning Models

Classification algorithms:

* K-Nearest Neighbors (KNN)
* Decision Tree
* Logistic Regression
* Multi-Layer Perceptron (MLP)
* Random Forest
* Extra Trees

Best performing model:

Random Forest

Performance:

* Accuracy: 57.25%
* F1-score: 55.86%

---

## Clustering Algorithms

* K-Means
* DBSCAN
* Agglomerative Clustering

Evaluation metrics:

* Silhouette Score
* Adjusted Rand Index (ARI)
* Normalized Mutual Information (NMI)

---

## Dashboard

The project includes an interactive Streamlit dashboard with:

* Live Emotion Dashboard
* Emotion Timeline Visualization
* Emotion Distribution Analysis
* Confidence Analysis
* Emotion Analyzer Page
* Real-time Prediction using the trained Random Forest model

---

## Installation

Install dependencies:

pip install -r requirements.txt

---

## Running the Dashboard

Run:

streamlit run dashboard/Live_Dashboard.py

---

## Authors

NeuroSense Project Team

UBT – Computer Science and Engineering
