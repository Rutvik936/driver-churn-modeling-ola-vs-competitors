# Driver Attrition Prediction (Ola, Uber, Rapido)

This is a web-based machine learning application designed to predict and analyze driver attrition for ride-sharing platforms like Ola, Uber, and Rapido. The project uses real-world data, performs preprocessing, visualizations, and predictive modeling using Random Forest and Gradient Boosting.

## 🔍 Features

- Driver attrition prediction using ML models (Random Forest, Gradient Boosting)
- Comparison of attrition trends across Ola, Uber, and Rapido
- Visual representation of attrition statistics using pie charts, bar plots, etc.
- CSV data processing and data cleaning
- Flask-based web interface with pages for:
  - Home
  - Data comparison
  - Prediction results


## ⚙️ Technologies Used

- Python
- Flask
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- HTML, CSS

## 🧠 Models Used

- **Random Forest Classifier** – Ensemble model for robust prediction
- **Gradient Boosting Classifier** – Boosted trees for improved accuracy

**Run the Flask App**

python app4.py
Visit http://127.0.0.1:5000/ in your browser.

# 🚕 Driver Attrition Prediction System

A machine learning-powered web application that predicts driver churn for major ride-hailing platforms (Ola, Uber, Rapido) using Flask.

![App Screenshot](static/screenshot.png) *(add screenshot path if available)*

## 📂 Project Structure

```plaintext
driver-attrition-prediction/
│
├── data/                    # Data storage
│   ├── Ola2.csv             # Processed datasets
│   ├── rapido.csv
│   ├── uber.csv
│
├── static/                  # Frontend assets
│   ├── dashboardstyle.css    # Comparison page styles
│   ├── styles.css           # Main stylesheet
│
├── templates/               # Flask templates
│   ├── index.html           # Landing page
│   ├── comparison.html      # Platform comparison
│   └── result.html          # Prediction results
│
├── app4.py                  # Flask application
├── ola.csv                  # Raw Ola dataset
├── uber.csv                 # Raw Uber dataset
├── rapido.csv               # Raw Rapido dataset
└── README.md                # Documentation
