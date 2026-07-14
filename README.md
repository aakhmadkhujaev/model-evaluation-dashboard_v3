# 📊 Model Evaluation Dashboard v3.0

A modular machine learning application for training, evaluating, and comparing supervised learning models.

The Model Evaluation Dashboard provides an end-to-end workflow for supervised machine learning, including dataset validation, preprocessing, model training, performance evaluation, report generation, visualization, and trained model serialization.

This project is part of the **ML Studio** series, a collection of modular machine learning tools designed to build a complete machine learning workflow from raw data to trained models.

---

## ✨ Features

- Load datasets from CSV files
- Validate datasets before training
- Automatic train/test split
- Automatic feature type detection
- Standard scaling for numerical features
- One-hot encoding for categorical features
- Support for Classification and Regression tasks
- Multiple machine learning algorithms
- Automatic performance evaluation
- Generate formatted evaluation reports
- Generate confusion matrix visualizations
- Save trained models using Joblib
- Modular and maintainable project architecture

---

# 🏗️ Project Architecture

```
Dataset
    │
    ▼
Dataset Validation
    │
    ▼
Train/Test Split
    │
    ▼
Preprocessing
 ├── Standard Scaling
 └── One-Hot Encoding
    │
    ▼
Model Training
    │
    ▼
Model Evaluation
    │
    ├── Performance Metrics
    ├── Report Generation
    ├── Visualization
    └── Model Saving
```

---

# 📁 Project Structure

```
Model_Evaluation_Dashboard_v3/
│
├── data/
│   └── engineered_cleaned_data.csv
│
├── outputs/
│   ├── figures/
│   ├── models/
│   └── reports/
│
├── src/
│   ├── config.py
│   ├── loader.py
│   ├── validator.py
│   ├── preprocessor.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── reporter.py
│   ├── visualizer.py
│   ├── model_saver.py
│   └── utils.py
│
├── tests/
├── main.py
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Model_Evaluation_Dashboard_v3.git
```

Navigate into the project

```bash
cd Model_Evaluation_Dashboard_v3
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

Run the application

```bash
python main.py
```

The dashboard will guide you through:

1. Selecting a dataset
2. Choosing the machine learning task
3. Selecting the target column
4. Choosing the machine learning model
5. Training and evaluating the model
6. Optionally generating reports
7. Optionally generating visualizations
8. Optionally saving the trained model

---

# 🤖 Supported Models

## Classification

- Logistic Regression
- Random Forest Classifier
- Support Vector Machine (SVC)

## Regression

- Linear Regression
- Random Forest Regressor
- Support Vector Regressor (SVR)

---

# 📈 Evaluation Metrics

## Classification

- Accuracy
- Precision
- Recall
- F1 Score

## Regression

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

# 📊 Generated Outputs

The application automatically creates organized outputs.

## Evaluation Report

```
outputs/reports/
```

Example

```
Support_Vector_Machine_20260714_104545.txt
```

---

## Confusion Matrix

```
outputs/figures/
```

Example

```
confusion_matrix_20260714_104556.png
```

---

## Saved Model

```
outputs/models/
```

Example

```
Support_Vector_Machine_20260714_104602.joblib
```

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Pathlib

---

# 🎯 Learning Objectives

This project demonstrates:

- Modular software architecture
- Clean code principles
- Machine learning workflow design
- Data preprocessing
- Model evaluation
- Performance reporting
- Model serialization
- Professional project organization

---

# 🔮 Future Improvements

- ROC Curve visualization
- Precision-Recall Curve
- Feature Importance visualization
- Regression residual plots
- Hyperparameter tuning
- Cross-validation
- Model comparison dashboard
- Pipeline serialization (Model + Preprocessor)
- Interactive command-line interface
- Export reports as PDF

---

# 🧩 ML Studio Roadmap

This project is the fifth module of the ML Studio ecosystem.

```
Dataset Inspector
        │
        ▼
EDA Report Generator
        │
        ▼
ML Preprocessing Assistant
        │
        ▼
Feature Engineering Assistant
        │
        ▼
✅ Model Evaluation Dashboard
        │
        ▼
Unsupervised Learning Studio
        │
        ▼
Deep Learning Studio
        │
        ▼
MLOps Studio
```

---

# 📄 License

This project is released under the MIT License.

---

# 👨‍💻 Author

**Abror Akhmadkhujaev**

Software Engineering Student

Machine Learning & AI Enthusiast

GitHub: https://github.com/aakhmadkhujaev

---

## ⭐ If you found this project useful, consider giving it a star!