Thank you! Here's the updated `README.md` with your clarification:

* ✅ Multiple models were used for anomaly detection.
* ✅ **XGBoost** was selected as the **final deployed model** based on performance.

---

# Bank Anomaly Prediction

An end-to-end project to detect **fraudulent or anomalous banking transactions** using multiple machine learning models, deployed via a clean and interactive **Streamlit web application** for real-time prediction.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-orange.svg)
![Machine Learning](https://img.shields.io/badge/Final%20Model-XGBoost-brightgreen.svg)

---

## 📌 Project Overview

This project focuses on detecting anomalies in banking transactions using machine learning. Several models were trained and compared, with the **XGBoost classifier** chosen for deployment due to its **superior performance** in identifying anomalies accurately.

---

## 🧠 Models Evaluated

The following models were tested for anomaly detection:

### 🧪 **Unsupervised Learning Models**

* **Isolation Forest**
* **Local Outlier Factor (LOF)**
* **One-Class SVM**

### ✅ **Supervised Learning Model**

* **XGBoost Classifier** *(Final deployed model)*

All models were evaluated using:

* Confusion Matrix
* Precision, Recall, F1-Score

📌 **XGBoost outperformed others and was selected for production deployment.**

---

## 🚀 Streamlit Web App

## 🔗 Live Demo

Check out the deployed app here:
👉 [**Bank Anomaly Prediction App**](https://bank-anomaly-prediction.streamlit.app/)

### Features:

* 🔘 Manual input of transaction data
* 📈 Real-time prediction of anomalies
* ✅ Deployed using the **best-performing model (XGBoost)**
* 🎨 Clean, modern UI with custom CSS and alert styling

### Run the App Locally

```bash
# Clone the repo
git clone https://github.com/furqank73/Bank_Anomaly_prediction.git
cd Bank_Anomaly_prediction

# Install required packages
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

---

## 🗂️ Project Structure

```
├── Bank_anomaly_prediction.ipynb   # EDA,feature engineering model training & evaluation
├── app.py                          # Streamlit app with prediction UI
├── best_model_xgboost.pkl/         # (Optional) Saved model files
├── requirements.txt                # List of dependencies
└── README.md                       # Project overview
```

---

## 📥 App Input Fields

* Transaction Amount
* Transaction Type (Credit/Debit)
* Channel (Online/ATM/Branch)
* Customer Age 
* Occupation
* Spend Ratio
* Login Attempts
* Time Since Last Transaction
* Account Balance

---

## 📸 Sample Screenshot

> *(![Screenshot](screencapture-bank-anomaly-prediction-streamlit-app-2025-07-06-23_04_18.png))*

---

## 🛠 Tech Stack

* Python
* Streamlit
* Scikit-learn
* XGBoost
* Pandas, NumPy
* Matplotlib, Seaborn

---

## 👨‍💻 Author

**M Furqan Khan**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin)](https://www.linkedin.com/in/furqan-khan-256798268/)
[![GitHub](https://img.shields.io/badge/GitHub-furqank73-black?logo=github)](https://github.com/furqank73)

---
