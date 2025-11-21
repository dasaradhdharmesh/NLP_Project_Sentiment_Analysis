
# **Sentiment Analysis Pro — NLP Web Application**

**AI-Powered Sentiment Analysis Web Application using Streamlit**

This project is a complete end-to-end Natural Language Processing (NLP) application built using Streamlit, designed to analyze sentiment from both single text inputs and large datasets.
It uses a Logistic Regression model with TF-IDF vectorization and includes data visualizations, intelligent preprocessing, and automated PDF reporting.

---

##  **Project Features**

### **Single Review Sentiment Analysis**

* Provides instant predictions for individual text inputs
* Displays sentiment class and confidence score
* Clean and minimal user interface designed for clarity

### **Bulk CSV Review Analysis**

* Automatically detects the text/review column
* Supports UTF-8 and Latin-1 encoded files
* Predicts sentiment for large volumes of data efficiently
* Computes sentiment distribution:
              * Positive
              * Neutral
              * Negative

* Generates visual insights through bar and pie charts
* Allows exporting of processed results as CSV
* Generates a complete PDF report with insights and sample predictions

### **Visual Analytics**

* Bar chart showing sentiment distribution
* Pie chart illustrating class proportions
* Metrics summarizing overall sentiment tendencies

### ** PDF Report Generation**

The automatically generated report includes:
* Summary of the analysis
* Sentiment counts and percentages
* Relevant charts
* Sample of processed rows
* Date, timestamp, and project details

## **Model Overview**

* **Algorithm:** Logistic Regression
* **Vectorization:** TF-IDF
* **Training Script:** 'train_and_save_models.py'
* **Model Artifacts:**
  * 'logistic_regression_model.pkl'
  * 'tfidf_vectorizer.pkl'


## **Project Structure**
'''
NLP_Project_Sentiment_Analysis
│
├── app_simple.py                # Streamlit web app
├── train_and_save_models.py     # Model training script
├── README.md                    # Project documentation
│
└── model/
    ├── logistic_regression_model.pkl
    └── tfidf_vectorizer.pkl
'''

---

## **How to Run the Project Locally**

### **1. Install dependencies**

```
pip install -r requirements.txt
```

### **2️. Run Streamlit app**

```
streamlit run app_simple.py
```

---

## **Requirements**

Here’s the recommended files:

```
streamlit
pandas
numpy
scikit-learn
joblib
matplotlib
seaborn
reportlab
```

---

## **Author**

**Dasaradh D**
B.Sc Computer Science
Data Science & Analytics

---


