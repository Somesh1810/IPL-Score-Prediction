# 🏏 IPL Score Prediction

A Machine Learning project that predicts the final score of an IPL innings using live match statistics such as batting team, bowling team, current score, overs, wickets, and recent performance.

This project demonstrates end-to-end ML workflow including data preprocessing, feature engineering, model building, evaluation, and deployment using Streamlit.

---

## 📌 Project Overview

The Indian Premier League (IPL) is one of the most competitive T20 cricket leagues in the world. Predicting the final innings score helps in:

- 📊 Match analytics
- 📈 Strategy building
- 🤖 Cricket data applications
- 🏏 Fantasy sports analysis

This model uses historical IPL match data to predict the final score at the end of 20 overs.

---

## 🚀 Features

- Data Cleaning & Preprocessing  
- Feature Engineering  
- Machine Learning Model Training  
- Model Evaluation (MAE / RMSE)  
- Interactive Streamlit Web Application  
- Real-time Score Prediction  

---

## 🛠️ Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Matplotlib / Seaborn  
- Streamlit  

---
IPL-score-prediction/
│
├── app.py # Streamlit application
├── model.pkl # Trained ML model
├── data.csv # IPL dataset
├── notebook.ipynb # Model training notebook
├── requirements.txt # Dependencies
└── README.md

---

## 📊 Input Features

The model takes the following inputs:

- Batting Team  
- Bowling Team  
- Current Runs  
- Wickets Fallen  
- Overs Completed  
- Runs in Last 5 Overs  
- Wickets in Last 5 Overs  

---

## 🧠 Machine Learning Model

The project uses:

- Linear Regression / Random Forest (update according to your actual model)
- Trained on historical IPL data
- Evaluated using:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)

---

## 📈 Project Workflow

1. Data Collection  
2. Data Cleaning  
3. Feature Engineering  
4. Train-Test Split  
5. Model Training  
6. Model Evaluation  
7. Deployment using Streamlit  

---

## ▶️ How to Run the Project

### 1️⃣ Clone the Repository

``
git clone https://github.com/Somesh1810/IPL-score-prediction.git
cd IPL-score-prediction``

## 📂 Project Structure
### Install Dependencies
pip install -r requirements.txt

### Run the Application
streamlit run app.py

## Live Demo

If deployed on Streamlit Cloud, add your live link here:

👉 https://your-app-link.streamlit.app

## Sample Prediction Output
Predicted Final Score: 178 - 185

## Future Improvements

Add Deep Learning models

Add Win Probability Prediction

Improve Feature Engineering

Deploy using Docker

Add Data Visualization Dashboard

## Author

Somesh M
Statistics & Data Science

GitHub: https://github.com/Somesh1810
