# modeling.py
import pandas as pd
import numpy as np
from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -------------------------
# ARIMA/SARIMA
# -------------------------
def fit_arima(train_series, seasonal=False, m=1):
    import pandas as pd
    from pmdarima import auto_arima

    # Ensure pandas Series
    train_series = train_series.copy()
    
    # Ensure datetime index
    if not isinstance(train_series.index, pd.DatetimeIndex):
        train_series.index = pd.to_datetime(train_series.index)
    
    # Set business-day frequency & forward-fill missing values
    train_series = train_series.asfreq('B').ffill()
    
    # Fit auto_arima
    model = auto_arima(
        train_series,
        seasonal=seasonal,
        m=m,
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore'
    )
    order = model.order
    seasonal_order = model.seasonal_order if seasonal else None
    return model, order, seasonal_order

def forecast_sarimax(train_series, order, seasonal_order=None, steps=1):
    """
    Fit SARIMAX on train_series and forecast `steps` ahead.
    Returns forecast array and fitted SARIMAX model.
    """
    train_series = train_series.asfreq('B').ffill()
    model = SARIMAX(
        train_series,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)
    forecast = model_fit.forecast(steps=steps)
    return forecast, model_fit

# -------------------------
# LSTM
# -------------------------
def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_lstm(model, X_train, y_train, epochs=50, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        validation_split=0.1, verbose=1)
    return model, history

def forecast_lstm(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred
