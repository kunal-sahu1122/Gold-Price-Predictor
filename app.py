import streamlit as st 
import yfinance as yf 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from xgboost import XGBRegressor 
from sklearn.metrics import mean_squared_error, r2_score 
from PIL import Image 
 
# 1. Fetch USD to INR conversion rate 
def fetch_usd_to_inr(): 
    try: 
        data = yf.download("INR=X", period="5d") 
        if data.empty or 'Close' not in data.columns: 
            st.error("USD to INR data not available.") 
            return None 
        usd_inr = data['Close'].dropna().iloc[-1] 
        return float(usd_inr) 
    except Exception as e: 
        st.error(f"Error fetching USD to INR rate: {e}") 
        return None 
 
# 2. Load gold price data 
def download_data(): 
    df = yf.download("GC=F", start="2005-01-01", end="2025-01-01") 
    df.reset_index(inplace=True) 
    df = df[['Date', 'Close']].dropna() 
    df['Year'] = df['Date'].dt.year 
    df['Month'] = df['Date'].dt.month 
    df['DayOfYear'] = df['Date'].dt.dayofyear 
    for i in range(1, 8): 
        df[f'lag_{i}'] = df['Close'].shift(i) 
    df.dropna(inplace=True) 
    return df 
 
# 3. Train model 
def train_model(df): 
    features = [f'lag_{i}' for i in range(1, 8)] + ['Year', 'Month', 'DayOfYear'] 
    train = df[df['Date'] < '2025-01-01'] 
    X_train = train[features] 
    y_train = train['Close'] 
    model = XGBRegressor(n_estimators=100, learning_rate=0.1) 
   
    model.fit(X_train, y_train) 
    return model, features, train 
 
# 4. Evaluate model 
def evaluate_model(model, features, df): 
    test = df[df['Date'] >= '2022-01-01'] 
    X_test = test[features] 
    y_test = test['Close'] 
    y_pred = model.predict(X_test) 
    mse = mean_squared_error(y_test, y_pred) 
    r2 = r2_score(y_test, y_pred) 
    return mse, r2, y_pred, y_test, test 
 
# 5. Forecast future prices 
def forecast_future(model, last_known, features): 
    future_preds = [] 
    future_dates = pd.date_range(start=last_known['Date'].iloc[-1] + pd.Timedelta(days=1), 
periods=365) 
 
    for future_date in future_dates: 
        input_row = [float(x) for x in last_known['Close'].values[-7:]] 
        year = future_date.year 
        month = future_date.month 
        doy = future_date.dayofyear 
        row = input_row + [year, month, doy] 
        row = np.array(row).reshape(1, -1) 
        pred = model.predict(row)[0] 
        future_preds.append(pred) 
 
        new_row = { 
            'Date': future_date, 
            'Close': pred, 
            'Year': year, 
            'Month': month, 
            'DayOfYear': doy 
        } 
        for i in range(1, 8): 
            new_row[f'lag_{i}'] = last_known['Close'].values[-i] 
        last_known = pd.concat([last_known, pd.DataFrame([new_row])], ignore_index=True) 
 
    return future_preds, future_dates 
 
# 6. Main app 
def main(): 
    st.set_page_config(layout="wide") 
    st.title("📈 Gold Price Prediction and Forecast (2005–2025)") 
 

    # Show image 
    image = Image.open(r"C:\Users\sahuk\OneDrive\Desktop\project4\ML\shutterstock
2480509399-2024-08-368b960cfc07a7fc6986b47f60f0159d-scaled.webp")  # Replace with your 
actual image path 
    st.image(image, caption='Gold Price Prediction 2024–2025', use_column_width=True) 
 
    # Chat input 
    st.subheader("💬 Ask something about gold prices") 
    user_input = st.text_input("Type your question here:") 
    if user_input: 
        st.write(f"You asked: {user_input}") 
        st.write("🔍 This prediction is based on historical trends and machine learning.") 
 
    # Load data 
    df = download_data() 
 
    # Get USD to INR rate 
    usd_inr = fetch_usd_to_inr() 
 
    if usd_inr is not None: 
        st.success(f"💱 Current USD to INR exchange rate: ₹{usd_inr:.2f}") 
 
        # Train and evaluate 
        model, features, train = train_model(df) 
        mse, r2, y_pred, y_test, test = evaluate_model(model, features, df) 
 
        st.write(f"📉 *Mean Squared Error (MSE)*: {mse:.2f}") 
        st.write(f"📈 *R² Score*: {r2:.2f}") 
 
        # Forecast future 
        last_known = df.iloc[-7:].copy() 
        future_preds, future_dates = forecast_future(model, last_known, features) 
 
        # Convert prices to INR per 10g 
        def usd_to_inr_10g(prices_usd): 
            return [(p * usd_inr / 31.1035) * 10 for p in prices_usd] 
 
        y_pred_inr = usd_to_inr_10g(y_pred) 
        future_preds_inr = usd_to_inr_10g(future_preds) 
        df_inr = df.copy() 
        df_inr['Close'] = (df_inr['Close'] * usd_inr / 31.1035) * 10 
 
        # Table 
        st.subheader("🧾 Forecast Table: Gold Prices in INR (per 10 grams)") 
        future_df = pd.DataFrame({ 
 

            'Date': future_dates, 
            'Predicted Price (USD)': future_preds, 
            'Predicted Price (INR per 10g)': future_preds_inr 
        }) 
        st.dataframe(future_df) 
 
        # Plot 
        st.subheader("📊 Gold Price Trends in INR (per 10 grams)") 
        fig, ax = plt.subplots(figsize=(14, 6)) 
        ax.plot(df_inr['Date'], df_inr['Close'], label='Actual Price (INR/10g)', color='blue') 
        ax.plot(test['Date'], y_pred_inr, label='Predicted Price (2022–2024)', color='orange') 
        ax.plot(future_dates, future_preds_inr, label='Forecast (2025)', color='green', linestyle='--') 
        ax.set_title("Gold Price Prediction and Forecast (₹/10g)") 
        ax.set_xlabel("Date") 
        ax.set_ylabel("Gold Price (₹ per 10g)") 
        ax.legend() 
        ax.grid(True) 
        st.pyplot(fig) 
 
    else: 
        st.warning("USD to INR exchange rate could not be fetched. Please check your internet or 
try again later.") 
 
if __name__ == "__main__":
    main()