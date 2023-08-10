# Develop simple time series model (arma) for Rwanda
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

pd.set_option("mode.copy_on_write", True)
# Ignore ValueWarning from TSA
import warnings
warnings.filterwarnings("ignore")


def get_time_series_prediction(train, test):
    # Sort data train by ID_LAT_LON and year_week
    train = train.sort_values(by=["ID_LAT_LON", "year_week"])
    train.set_index("year_week", inplace=True)

    test = test.sort_values(by=["ID_LAT_LON", "year_week"])
    test = test["ID_LAT_LON_YEAR_WEEK"]

    # Estimate model
    model = ARIMA(train["emission"], order=(1,3,1))
    model_fit = model.fit()

    # Make prediction
    yhat = model_fit.predict(1, len(test))

    # Covert pd.Series to pd.DataFrame
    yhat = pd.DataFrame(yhat)
    # Rename predicted_mean to emission
    yhat = yhat.rename(columns={"predicted_mean": "emission"})
    yhat = yhat.reset_index(drop=True)
    test = test.reset_index(drop=True)
    yhat = pd.concat([test, yhat], axis=1)
    return yhat
