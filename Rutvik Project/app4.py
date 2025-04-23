import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI environments

# Import necessary libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import os
from io import BytesIO
import base64
from collections import Counter

# Scikit-learn and ML-related libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import SMOTE

app = Flask(__name__)

static_folder = "static"
if not os.path.exists(static_folder):
    os.makedirs(static_folder)

# Load dataset (only first 100 rows)
data_file = "ola.csv"

def generate_plots():
    df = pd.read_csv(data_file).head(100)  # Take first 100 entries

    # Create 'Attrition' column based on 'LastWorkingDate'
    df["Attrition"] = df["LastWorkingDate"].notna().map({True: 1, False: 0})

    # 1️⃣ Pie Chart: Attrition Distribution
    plt.figure(figsize=(6, 6))
    df["Attrition"].value_counts().plot(kind="pie", autopct="%1.1f%%", colors=["lightcoral", "lightgreen"])
    plt.title("Driver Attrition Distribution")
    plt.ylabel("")
    plt.savefig(os.path.join(static_folder, "pie_chart.png"))
    plt.close()

    # 2️⃣ Bar Chart: Attrition by City
    plt.figure(figsize=(8, 5))
    df[df["Attrition"] == 1]["City"].value_counts().plot(kind="bar", color="skyblue")
    plt.title("Attrition by City")
    plt.xlabel("City")
    plt.ylabel("Number of Drivers")
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(static_folder, "bar_chart.png"))
    plt.close()

    # 3️⃣ Box Plot: Income Distribution by Attrition
    plt.figure(figsize=(7, 5))
    sns.boxplot(x=df["Attrition"], y=df["Income"], palette=["lightcoral", "lightgreen"])
    plt.title("Income Distribution by Attrition")
    plt.xticks([0, 1], ["Stayed", "Left"])
    plt.savefig(os.path.join(static_folder, "box_plot.png"))
    plt.close()

    # 4️⃣ Bivariate Analysis (Pairplot for numerical features)
    numeric_features = ["Age", "Income", "Total Business Value", "Quarterly Rating"]
    sns.pairplot(df[numeric_features + ["Attrition"]], hue="Attrition", palette=["red", "green"])
    plt.savefig(os.path.join(static_folder, "pairplot.png"))
    plt.close()

    # 5️⃣ Correlation Matrix
    plt.figure(figsize=(8, 6))
    corr = df[numeric_features + ["Attrition"]].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.savefig(os.path.join(static_folder, "correlation_matrix.png"))
    plt.close()

    # 6️⃣ ROC Curve using Logistic Regression
    X = df[["Age", "Income", "Total Business Value", "Quarterly Rating"]].fillna(0)
    y = df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="blue", label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "r--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(static_folder, "roc_curve.png"))
    plt.close()

# ======================== BRANDING CONFIG ========================
PLATFORM_DATA = {
    "Ola": {"color": "#0B5AB0", "logo": "ola-logo.png"},
    "Uber": {"color": "#000000", "logo": "uber-logo.png"},
    "Rapido": {"color": "#FF6B00", "logo": "rapido-logo.png"}
}

# ======================== DATA LOADING ========================
def load_data(filename, sample_size=None):
    """Load data from CSV file with optional sampling"""
    data = {}
    with open(f'data/{filename}', 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if sample_size and i >= sample_size:
                break
            driver_id = int(row['driver_id'])
            # Ensure driver_id is unique
            data[driver_id] = {
                "attrition_reason": row['attrition_reason'] if row['attrition_reason'] else None,
                "attrition_status": row['attrition_status'].lower() == 'true'
            }
    return data

# Load datasets (50 entries for Ola, full for others)
ola_attrition = load_data('ola2.csv', sample_size=50)
uber_attrition = load_data('uber.csv')
rapido_attrition = load_data('rapido.csv')

# ======================== HELPER FUNCTIONS ========================
def count_reasons(data):
    """Count attrition reasons from a dataset"""
    reasons = [driver["attrition_reason"] for driver in data.values() 
              if driver["attrition_status"] and driver["attrition_reason"]]
    return dict(Counter(reasons))

def generate_chart(platforms_data):
    """Generate comparison bar chart"""
    plt.figure(figsize=(14, 7))
    
    # Get all unique reasons
    all_reasons = set()
    for counts in platforms_data.values():
        all_reasons.update(counts.keys())
    reasons = sorted(all_reasons)
    
    # Plot settings
    bar_width = 0.25
    index = range(len(reasons))
    
    # Plot bars for each platform
    for i, (platform, counts) in enumerate(platforms_data.items()):
        platform_counts = [counts.get(reason, 0) for reason in reasons]
        plt.bar(
            [x + i * bar_width for x in index],
            platform_counts,
            width=bar_width,
            color=PLATFORM_DATA[platform]["color"],
            label=platform,
            edgecolor='white'
        )
    
    # Chart styling
    plt.title('Driver Attrition Reasons Comparison', fontsize=16, pad=20)
    plt.xlabel('Reasons', fontsize=12)
    plt.ylabel('Number of Drivers', fontsize=12)
    plt.xticks([x + bar_width for x in index], reasons, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    
    # Save to bytes
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Load and preprocess dataset
df = pd.read_csv('https://d2beiqkhq929f0.cloudfront.net/public_assets/assets/000/002/492/original/ola_driver_scaler.csv')
df.rename(columns={'MMM-YY': 'Reporting Date', 'Dateofjoining': 'Date of Joining', 'LastWorkingDate': 'Last Working Date'}, inplace=True)
df['Reporting Date'] = pd.to_datetime(df['Reporting Date'], format='%d/%m/%y')
df['Date of Joining'] = pd.to_datetime(df['Date of Joining'], format='%d/%m/%y')
df['Last Working Date'] = pd.to_datetime(df['Last Working Date'], format='%d/%m/%y')

# Feature Engineering
df['Quarterly_Rating_Change'] = df.groupby('Driver_ID')['Quarterly Rating'].diff().fillna(0).apply(lambda x: 1 if x > 0 else 0)
df['Income_Change'] = df.groupby('Driver_ID')['Income'].diff().fillna(0).apply(lambda x: 1 if x > 0 else 0)
df['Target'] = df['Last Working Date'].apply(lambda x: 1 if pd.notnull(x) else 0)

# Aggregation
df_agg = df.groupby('Driver_ID').agg({
    'Age': 'max',
    'Gender': 'max',
    'City': 'first',
    'Education_Level': 'first',
    'Income': 'mean',
    'Date of Joining': 'first',
    'Last Working Date': 'first',
    'Joining Designation': 'first',
    'Grade': 'mean',
    'Total Business Value': 'sum',
    'Quarterly Rating': 'mean'
}).reset_index()

df_features = df.groupby('Driver_ID').agg({
    'Quarterly_Rating_Change': 'max',
    'Income_Change': 'max',
    'Target': 'max'
}).reset_index()

df_agg = df_agg.merge(df_features, on='Driver_ID', how='left')

# Encoding and Scaling
# Handle unknown categories with 'ignore'
encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
encoded_features = encoder.fit_transform(df_agg[['Education_Level']])
encoded_feature_names = encoder.get_feature_names_out()

scaler = StandardScaler()
scaled_features = scaler.fit_transform(df_agg[['Age', 'Income', 'Total Business Value', 'Grade']])

X = np.hstack((scaled_features, encoded_features, df_agg[['Quarterly_Rating_Change', 'Income_Change']].values))
y = df_agg['Target'].values

# SMOTE for Class Imbalance
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)

# Random Forest with Hyperparameter Tuning
rf = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='roc_auc')
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_

# Gradient Boosting Model
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)

# ======================== ROUTES ========================

@app.route("/")
def index():
    generate_plots()  # Generate plots before rendering
    return render_template("index.html")


@app.route('/comparison')
def comparison():
    # Process data
    platforms = {
        "Ola": count_reasons(ola_attrition),
        "Uber": count_reasons(uber_attrition),
        "Rapido": count_reasons(rapido_attrition)
    }
    
    # Generate visualization
    chart = generate_chart(platforms)
    
    # Calculate attrition rates
    stats = {}
    for name, data in [("Ola", ola_attrition), ("Uber", uber_attrition), ("Rapido", rapido_attrition)]:
        left = sum(1 for d in data.values() if d["attrition_status"])
        total = len(data)
        stats[name] = {
            "attrition_rate": f"{(left/total)*100:.1f}%",
            "sample_size": total,
            "top_reason": max(platforms[name].items(), key=lambda x: x[1])[0] if platforms[name] else "N/A"
        }
    print(f"Generated stats: {stats}")
    
    return render_template(
        'comparison.html',
        plot_url=chart,
        platforms=platforms,
        stats=stats,  # Explicitly passed
        PLATFORM_DATA=PLATFORM_DATA
    )
# Route: Home
@app.route('/result')
def result():
    return render_template('result.html', result={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {
            'Age': int(request.form['Age']),
            'Income': int(request.form['Income']),
            'Education_Level': int(request.form['Education_Level']),
            'Total Business Value': int(request.form['Total_Business_Value']),
            'Grade': int(request.form['Grade']),
            'Quarterly_Rating_Change': int(request.form['Quarterly_Rating_Change']),
            'Income_Change': int(request.form['Income_Change'])
        }

        input_df = pd.DataFrame([input_data])
        input_encoded = encoder.transform(input_df[['Education_Level']])
        input_encoded_df = pd.DataFrame(input_encoded, columns=encoded_feature_names)

        input_scaled = scaler.transform(input_df[['Age', 'Income', 'Total Business Value', 'Grade']])
        input_final = np.hstack((input_scaled, input_encoded_df.values, input_df[['Quarterly_Rating_Change', 'Income_Change']].values))

        rf_pred = best_rf.predict(input_final)[0]
        rf_prob = best_rf.predict_proba(input_final)[0][1]

        gb_pred = gb.predict(input_final)[0]
        gb_prob = gb.predict_proba(input_final)[0][1]

        result = {
            'Random Forest Prediction': 'Driver will leave' if rf_pred == 1 else 'Driver will stay',
            'Random Forest Probability': f"{rf_prob * 100:.2f}%",
            'Gradient Boosting Prediction': 'Driver will leave' if gb_pred == 1 else 'Driver will stay',
            'Gradient Boosting Probability': f"{gb_prob * 100:.2f}%"
        }

        return render_template('result.html', result=result)

    except Exception as e:
        return f"An error occurred: {e}"

    
if __name__ == '__main__':
    app.run(debug=True)