# 📊 Sales Forecasting Model using Prophet

A complete end-to-end **time-series forecasting system** built using **Python** and **Meta’s Prophet** library.  
This project loads sales data, performs EDA, trains a forecasting model, evaluates accuracy using RMSLE, and generates visual insights.  
It also saves the trained model for reuse — making it ready for deployment or integration into a Streamlit dashboard.

---

## 🚀 Features

### 🔹 1. Automated Data Loading
- Reads CSV files using Pandas  
- Auto-detects date columns (`data`, `date`, `ds`)  
- Auto-detects sales/target columns (`venda`, `sales`, `y`)  
- Validates file paths and handles errors

### 🔹 2. Exploratory Data Analysis (EDA)
Generates insights through:
- DataFrame info and summary statistics  
- Line plots for:
  - Sales
  - Stock
  - Price  
- Helps visualize trends, seasonality, and outliers

### 🔹 3. Data Preprocessing
- Converts date column to datetime  
- Renames columns for Prophet (`ds`, `y`)  
- Sorts chronologically  
- Keeps data clean and model-ready

### 🔹 4. Prophet Forecasting
- Trains a Meta Prophet model  
- Creates future timeline (default: **365 days**)  
- Predicts:
  - `yhat` (forecast)
  - `yhat_lower`
  - `yhat_upper`  
- Produces full forecast DataFrame

### 🔹 5. Model Evaluation
- Calculates **RMSLE** (Root Mean Squared Logarithmic Error)  
- Evaluates accuracy on overlapping actual vs predicted values  
- Ensures invalid values are excluded

### 🔹 6. Visualizations
Automatically generates:
- Forecast plot  
- Trend component  
- Weekly seasonality  
- Yearly seasonality (if present)

### 🔹 7. Model Saving
Trained model is saved as:

joblib.dump(model, "prophet_model.pkl")

So it can be reused without retraining.

---

## 🛠️ Tech Stack

| Category | Tools |
|---------|-------|
| **Language** | Python |
| **Data Analysis** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Forecasting Model** | Prophet |
| **ML Evaluation** | Scikit-Learn |
| **Model Storage** | Joblib |
| **CLI Support** | Argparse |
| **IDE** | PyCharm |

---
