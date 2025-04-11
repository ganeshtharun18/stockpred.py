import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import VotingRegressor
import matplotlib.pyplot as plt

# Function to fetch stock data from yfinance
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    data = stock.history(period="1y")
    data.reset_index(inplace=True)
    data = data[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
    return data

# Function to preprocess data
def preprocess_data(data):
    data = data.dropna()
    scaler = StandardScaler()
    features = data.drop(columns=['Date', 'Close'])
    target = data['Close']
    features_scaled = scaler.fit_transform(features)
    return features_scaled, target

# PCA function
def apply_pca(features, n_components):
    pca = PCA(n_components=n_components)
    features_pca = pca.fit_transform(features)
    return features_pca

# Streamlit app
def main():
    st.title("Stock Price Prediction for Next 10 Days")

    # User input for stock ticker
    st.sidebar.header("Stock Selection")
    stock_ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL")

    if stock_ticker:
        st.write(f"### Stock Data for {stock_ticker}")

        try:
            # Fetch and display stock data
            data = fetch_stock_data(stock_ticker)
            st.write("### Raw Data")
            st.dataframe(data.head())

            # Preprocess data
            features, target = preprocess_data(data)

            # Ensure PCA components don't exceed the data's dimensions
            n_components = min(5, features.shape[1])
            features_pca = apply_pca(features, n_components=n_components)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(features_pca, target, test_size=0.2, random_state=42)

            # Train models
            lr_model = LinearRegression()
            rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

            lr_model.fit(X_train, y_train)
            rf_model.fit(X_train, y_train)

            # Ensemble model
            ensemble_model = VotingRegressor([('lr', lr_model), ('rf', rf_model)])
            ensemble_model.fit(X_train, y_train)

            # Predictions
            lr_pred = lr_model.predict(X_test)
            rf_pred = rf_model.predict(X_test)
            ensemble_pred = ensemble_model.predict(X_test)

            # Function to calculate accuracy
            def calculate_accuracy(y_true, y_pred, tolerance=0.01):
                within_tolerance = np.abs(y_true - y_pred) <= (tolerance * y_true)
                accuracy = np.mean(within_tolerance) * 100  # Convert to percentage
                return accuracy

            # Evaluation function
            def evaluate_model(name, y_true, y_pred):
                mse = mean_squared_error(y_true, y_pred)
                r2 = r2_score(y_true, y_pred)
                accuracy = calculate_accuracy(y_true, y_pred)
                return mse, r2, accuracy

            # Evaluate models
            lr_mse, lr_r2, lr_accuracy = evaluate_model('Linear Regression', y_test, lr_pred)
            rf_mse, rf_r2, rf_accuracy = evaluate_model('Random Forest', y_test, rf_pred)
            ensemble_mse, ensemble_r2, ensemble_accuracy = evaluate_model('Ensemble', y_test, ensemble_pred)

            # Display performance
            st.write("### Model Performance")
            performance = pd.DataFrame({
                'Model': ['Linear Regression', 'Random Forest', 'Ensemble'],
                'MSE': [lr_mse, rf_mse, ensemble_mse],
                'R2 Score': [lr_r2, rf_r2, ensemble_r2],
                'Accuracy (%)': [lr_accuracy, rf_accuracy, ensemble_accuracy]
            })
            st.dataframe(performance)

            # Future prediction (last 10 days)
            st.write("### Predicting Next 10 Days")
            recent_data = features_pca[-10:]
            if len(recent_data) < 10:
                st.warning("Not enough data for a full 10-day prediction.")
                return

            future_pred = ensemble_model.predict(recent_data)
            future_dates = pd.date_range(start=data['Date'].iloc[-1], periods=10).strftime('%Y-%m-%d')
            future_df = pd.DataFrame({"Date": future_dates, "Predicted Close": future_pred})
            st.write(future_df)

            # Comparison plots
            fig, ax = plt.subplots()
            ax.plot(y_test.values[:50], label='True Prices', marker='o')
            ax.plot(lr_pred[:50], label='Linear Regression', marker='x')
            ax.plot(rf_pred[:50], label='Random Forest', marker='s')
            ax.plot(ensemble_pred[:50], label='Ensemble', marker='d')
            ax.legend()
            ax.set_title("Model Comparison (First 50 Test Samples)")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error fetching data for {stock_ticker}: {e}")

if __name__ == "__main__":
    main()
