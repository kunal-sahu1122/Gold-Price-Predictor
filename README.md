# Gold Price Predictor

## Overview
Gold Price Predictor is a Machine Learning-based web application that predicts and forecasts gold prices using historical gold market data. The application fetches real-time gold prices and USD to INR exchange rates, then uses an XGBoost Regression model to generate predictions and future forecasts.

## Features
- Real-time gold price data using Yahoo Finance
- Live USD to INR exchange rate conversion
- Gold price prediction using XGBoost Regressor
- Future price forecasting
- Interactive Streamlit dashboard
- Gold price visualization with graphs and tables

## Tech Stack
- Python
- Streamlit
- Pandas
- NumPy
- Matplotlib
- XGBoost
- Scikit-Learn
- Yahoo Finance API

## Machine Learning Model
- Algorithm: XGBoost Regressor
- Features Used:
  - Previous 7-day gold prices (Lag Features)
  - Year
  - Month
  - Day of Year

## Project Workflow
1. Fetch historical gold price data from Yahoo Finance.
2. Preprocess and create lag features.
3. Train the XGBoost model.
4. Evaluate model performance using MSE and R² Score.
5. Forecast future gold prices.
6. Convert predicted prices from USD to INR.
7. Display results using Streamlit.

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Gold-Price-Predictor.git
cd Gold-Price-Predictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Project Structure

```text
Gold-Price-Predictor/
│
├── app.py
├── requirements.txt
├── README.md
└── images/
```

## Future Improvements
- Integration with Deep Learning models (LSTM)
- Multi-factor gold price prediction
- Deployment on cloud platforms
- Enhanced forecasting accuracy

## Author
Kunal Sahu
