# Lesson 3: Prophet & Modern Approaches 🚀

**Module 7: Time Series & Forecasting | Lesson 3 of 4**

Master Prophet - Facebook's forecasting tool for business time series!

---

## Why Prophet?

**ARIMA limitations:**
- Requires stationarity
- Manual parameter tuning
- Struggles with missing data/outliers

**Prophet advantages:**
- Handles seasonality automatically
- Robust to missing data & outliers
- Interpretable components
- Works well for business forecasting

---

## 1. Prophet Basics 📊

```python
from prophet import Prophet
import pandas as pd

# Prophet requires specific column names
df_prophet = pd.DataFrame({
    'ds': df.index,  # Dates
    'y': df['value'].values  # Values
})

# Create and fit model
model = Prophet()
model.fit(df_prophet)

# Make future dataframe
future = model.make_future_dataframe(periods=365)  # 365 days ahead

# Predict
forecast = model.predict(future)

# Plot
fig = model.plot(forecast)
plt.show()

# Plot components
fig = model.plot_components(forecast)
plt.show()
```

---

## 2. Handling Seasonality 📆

```python
# Custom seasonality
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False
)

# Add custom seasonality (e.g., monthly)
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

model.fit(df_prophet)
forecast = model.predict(future)
```

---

## 3. Holidays & Special Events 🎉

```python
# Define holidays
holidays = pd.DataFrame({
    'holiday': 'black_friday',
    'ds': pd.to_datetime(['2020-11-27', '2021-11-26', '2022-11-25']),
    'lower_window': 0,
    'upper_window': 1,
})

model = Prophet(holidays=holidays)
model.fit(df_prophet)

# Christmas
christmas = pd.DataFrame({
    'holiday': 'christmas',
    'ds': pd.to_datetime(['2020-12-25', '2021-12-25', '2022-12-25']),
    'lower_window': -7,  # Week before
    'upper_window': 7,   # Week after
})

model = Prophet(holidays=pd.concat([holidays, christmas]))
model.fit(df_prophet)
```

---

## 4. Evaluation 📈

```python
from prophet.diagnostics import cross_validation, performance_metrics

# Cross-validation
df_cv = cross_validation(
    model,
    initial='730 days',  # Initial training period
    period='180 days',   # Spacing between cutoff dates
    horizon='365 days'   # Forecast horizon
)

# Performance metrics
df_metrics = performance_metrics(df_cv)
print(df_metrics.head())

# MAPE, MAE, MSE, RMSE
```

---

## 5. Complete Example: Sales Forecasting 💰

```python
# Load sales data
sales = pd.DataFrame({
    'ds': pd.date_range('2019-01-01', '2022-12-31', freq='D'),
    'y': np.random.randint(1000, 5000, 1461)  # Dummy sales
})

# Add trend and seasonality
sales['y'] = sales['y'] + np.arange(len(sales)) * 0.5  # Trend
sales['y'] = sales['y'] + 500 * np.sin(2 * np.pi * np.arange(len(sales)) / 365)  # Yearly

# Train/test split
train = sales[sales['ds'] < '2022-01-01']
test = sales[sales['ds'] >= '2022-01-01']

# Fit Prophet
model = Prophet(
    changepoint_prior_scale=0.05,  # Trend flexibility
    seasonality_prior_scale=10,     # Seasonality strength
    seasonality_mode='multiplicative'  # or 'additive'
)

model.fit(train)

# Forecast
future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)

# Evaluate
test_forecast = forecast[forecast['ds'] >= '2022-01-01']

mae = mean_absolute_error(test['y'], test_forecast['yhat'])
rmse = np.sqrt(mean_squared_error(test['y'], test_forecast['yhat']))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Plot
plt.figure(figsize=(12, 4))
plt.plot(train['ds'], train['y'], label='Train')
plt.plot(test['ds'], test['y'], label='Test')
plt.plot(test_forecast['ds'], test_forecast['yhat'], label='Forecast', linestyle='--')
plt.fill_between(test_forecast['ds'],
                test_forecast['yhat_lower'],
                test_forecast['yhat_upper'],
                alpha=0.3, label='Confidence')
plt.legend()
plt.show()
```

---

## 6. Hyperparameter Tuning 🎛️

```python
from itertools import product

# Grid search
param_grid = {
    'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
}

# Generate all combinations
all_params = [dict(zip(param_grid.keys(), v))
             for v in product(*param_grid.values())]

maes = []

for params in all_params:
    model = Prophet(**params).fit(train)
    forecast = model.predict(future)
    test_forecast = forecast[forecast['ds'] >= '2022-01-01']

    mae = mean_absolute_error(test['y'], test_forecast['yhat'])
    maes.append(mae)

# Best parameters
best_params = all_params[np.argmin(maes)]
print(f"Best params: {best_params}")
print(f"Best MAE: {min(maes):.2f}")
```

---

## Quick Reference 📖

**Prophet Template:**
```python
from prophet import Prophet

# Prepare data
df = pd.DataFrame({'ds': dates, 'y': values})

# Create model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    seasonality_mode='additive',  # or 'multiplicative'
    changepoint_prior_scale=0.05
)

# Add custom components
model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
model.add_country_holidays(country_name='US')

# Fit
model.fit(df)

# Forecast
future = model.make_future_dataframe(periods=365)
forecast = model.predict(future)

# Visualize
model.plot(forecast)
model.plot_components(forecast)
```

**Key Parameters:**
- `changepoint_prior_scale`: Trend flexibility (default: 0.05)
- `seasonality_prior_scale`: Seasonality strength (default: 10)
- `seasonality_mode`: 'additive' or 'multiplicative'

---

## Key Takeaways 💡

1. **Prophet** = production-ready forecasting
2. Handles **missing data & outliers** automatically
3. **Interpretable components** (trend, seasonality)
4. Great for **business forecasting**
5. **Easy to add** holidays & events
6. **Cross-validation** built-in

---

**Next:** [Lesson 4 - Deep Learning for Time Series →](Lesson%204%20-%20Deep%20Learning%20for%20Time%20Series.md)
