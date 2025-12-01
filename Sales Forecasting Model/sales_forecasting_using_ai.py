"""
Clean, runnable Python script converted from the Colab notebook.

Usage:
    - Edit the default FILE_PATH to point to your CSV if needed, or pass --file argument.
    - Install required packages: pandas, numpy, matplotlib, seaborn, prophet, scikit-learn, joblib
        pip install pandas numpy matplotlib seaborn prophet scikit-learn joblib

This script will:
  - load the CSV
  - run basic EDA
  - prepare data for Prophet
  - train a Prophet model
  - produce a forecast and compute RMSLE
  - save the trained model to prophet_model.pkl

"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.metrics import mean_squared_log_error
import joblib


def load_data(file_path: str) -> pd.DataFrame:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"CSV file not found at: {file_path}")

    df = pd.read_csv(file_path)
    return df


def prepare_dataframe(df: pd.DataFrame, date_col_candidates=None, target_col_candidates=None) -> pd.DataFrame:
    if date_col_candidates is None:
        date_col_candidates = ['data', 'date', 'ds']
    if target_col_candidates is None:
        target_col_candidates = ['venda', 'sales', 'y']

    # find date column
    date_col = None
    for c in date_col_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"No date column found. Tried: {date_col_candidates}. Available columns: {list(df.columns)}")

    # find target column
    target_col = None
    for c in target_col_candidates:
        if c in df.columns:
            target_col = c
            break
    if target_col is None:
        raise ValueError(f"No target column (sales) found. Tried: {target_col_candidates}. Available columns: {list(df.columns)}")

    # ensure datetime
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    if df[date_col].isna().any():
        raise ValueError(f"Some dates could not be parsed in column '{date_col}'. Check the CSV format.")

    # rename for Prophet
    df = df.rename(columns={date_col: 'ds', target_col: 'y'})

    # sort by date
    df = df.sort_values('ds').reset_index(drop=True)

    # keep only ds and y for modeling (but keep original columns in a copy if needed)
    df_model = df[['ds', 'y']].copy()

    return df_model, df


def plot_time_series(df: pd.DataFrame, original_df: pd.DataFrame = None):
    sns.set_style('whitegrid')
    fig, axes = plt.subplots(3, 1, figsize=(14, 16), sharex=True)

    # venda / y
    sns.lineplot(x='ds', y='y', data=df, ax=axes[0])
    axes[0].set_title('Target (y) over time')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('y')

    # estoque if present
    if original_df is not None and 'estoque' in original_df.columns:
        sns.lineplot(x='ds', y='estoque', data=original_df, ax=axes[1])
        axes[1].set_title('Estoque over time')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('estoque')
    else:
        axes[1].axis('off')

    # preco if present
    if original_df is not None and 'preco' in original_df.columns:
        sns.lineplot(x='ds', y='preco', data=original_df, ax=axes[2])
        axes[2].set_title('Preco over time')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel('preco')
    else:
        axes[2].axis('off')

    plt.tight_layout()
    plt.show()


def train_and_forecast(df_model: pd.DataFrame, periods: int = 365) -> (Prophet, pd.DataFrame):
    model = Prophet()
    model.fit(df_model)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return model, forecast


def compute_rmsle(df_actual_pred: pd.DataFrame) -> float:
    # filter positive values
    valid = df_actual_pred[(df_actual_pred['y'] > 0) & (df_actual_pred['yhat'] > 0)]
    if len(valid) == 0:
        raise ValueError('No positive actual/predicted pairs available to compute RMSLE.')
    rmsle = np.sqrt(mean_squared_log_error(valid['y'], valid['yhat']))
    return rmsle


def main(file_path: str, periods: int = 365, save_model: bool = True):
    print(f"Loading data from: {file_path}")
    df_raw = load_data(file_path)
    print("Data loaded. Columns:", df_raw.columns.tolist())

    df_model, df_full = prepare_dataframe(df_raw)
    print("Prepared DataFrame for Prophet. Rows:", len(df_model))
    print(df_model.head())

    # Visualize (optional)
    try:
        plot_time_series(df_model, original_df=df_full)
    except Exception as e:
        print("Plotting skipped due to:", e)

    print("Training Prophet model...")
    model, forecast = train_and_forecast(df_model, periods=periods)
    print("Forecast generated. Sample:")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())

    # Merge to compute RMSLE on historical portion
    df_cv = pd.merge(df_model, forecast[['ds', 'yhat']], on='ds', how='inner')
    try:
        rmsle = compute_rmsle(df_cv)
        print(f"RMSLE on overlapping historical dates: {rmsle:.4f}")
    except Exception as e:
        print("Could not compute RMSLE:", e)

    # Plot forecast using Prophet's built-in plot
    try:
        fig1 = model.plot(forecast)
        plt.show()
        fig2 = model.plot_components(forecast)
        plt.show()
    except Exception as e:
        print("Could not plot Prophet results:", e)

    # Save model
    if save_model:
        out_model_path = os.path.join(os.path.dirname(file_path), 'prophet_model.pkl')
        joblib.dump(model, out_model_path)
        print(f"Trained model saved to: {out_model_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sales forecasting using Prophet')
    parser.add_argument('--file', type=str, default=r"C:\Users\ASUS\OneDrive\Desktop\Sales Forecasting Model\mock_kaggle.csv",
                        help='Path to the CSV file')
    parser.add_argument('--periods', type=int, default=365, help='Number of future periods to forecast')
    parser.add_argument('--no-save', dest='save', action='store_false', help='Do not save the trained model')
    args = parser.parse_args()

    main(file_path=args.file, periods=args.periods, save_model=args.save)
