# FUTURE_ML_01
A forward-looking machine learning project exploring predictive analytics and intelligent automation. This project focuses on building scalable ML models using real-world datasets to forecast trends, optimize decisions, and enable smarter systems across various industries. 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet

# Simulate daily sales data
np.random.seed(42)
dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq='D')
sales = 500 + 30*np.sin(2 * np.pi * dates.dayofyear / 365) + np.random.normal(0, 25, len(dates))
df = pd.DataFrame({'ds': dates, 'y': sales})

# Plot sales data
plt.figure(figsize=(14, 5))
plt.plot(df['ds'], df['y'], color='blue', label='Sales')
plt.title("Daily Sales Data (2023)")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.grid(True)
plt.legend()
plt.show()

# Fit Prophet model
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Plot forecast
model.plot(forecast)
plt.title("Sales Forecast")
plt.show()

# Plot components
model.plot_components(forecast)
plt.show()


