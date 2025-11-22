# Lesson 2: ARIMA & Statistical Methods 📈

**Module 7: Time Series & Forecasting | Lesson 2 of 4**

Master ARIMA - the classic time series forecasting method!

---

## ARIMA Components

**ARIMA(p, d, q):**
- **AR(p):** AutoRegressive (past values)
- **I(d):** Integrated (differencing)
- **MA(q):** Moving Average (past errors)

---

## 1. AR (AutoRegressive) 🔄

```python
# AR(p): y_t = c + φ₁y_{t-1} + φ₂y_{t-2} + ... + φ_p y_{t-p} + ε_t

from statsmodels.tsa.ar_model import AutoReg

# Fit AR model
model = AutoReg(train['value'], lags=5)
model_fit = model.fit()

# Forecast
forecast = model_fit.predict(start=len(train), end=len(train)+len(test)-1)

# Evaluate
mae = mean_absolute_error(test['value'], forecast)
print(f"AR(5) MAE: {mae:.2f}")
```

---

## 2. MA (Moving Average) 📊

```python
# MA(q): y_t = μ + ε_t + θ₁ε_{t-1} + ... + θ_q ε_{t-q}

from statsmodels.tsa.arima.model import ARIMA

# MA(q) is ARIMA(0, 0, q)
model = ARIMA(train['value'], order=(0, 0, 3))
model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test))

mae = mean_absolute_error(test['value'], forecast)
print(f"MA(3) MAE: {mae:.2f}")
```

---

## 3. ARIMA Model 🎯

```python
# Full ARIMA(p, d, q)
model = ARIMA(train['value'], order=(2, 1, 2))
model_fit = model.fit()

# Summary
print(model_fit.summary())

# Forecast
forecast = model_fit.forecast(steps=len(test))

# Evaluate
mae = mean_absolute_error(test['value'], forecast)
rmse = np.sqrt(mean_squared_error(test['value'], forecast))

print(f"ARIMA(2,1,2) MAE: {mae:.2f}")
print(f"ARIMA(2,1,2) RMSE: {rmse:.2f}")

# Plot
plt.figure(figsize=(12, 4))
plt.plot(train.index, train['value'], label='Train')
plt.plot(test.index, test['value'], label='Test')
plt.plot(test.index, forecast, label='Forecast', linestyle='--')
plt.legend()
plt.show()
```

---

## 4. Auto ARIMA (Automatic Parameter Selection) ⭐

```python
from pmdarima import auto_arima

# Automatically find best parameters
model = auto_arima(
    train['value'],
    start_p=0, start_q=0,
    max_p=5, max_q=5,
    seasonal=False,
    d=None,  # Auto-determine differencing
    trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True
)

print(model.summary())

# Forecast
forecast = model.predict(n_periods=len(test))

mae = mean_absolute_error(test['value'], forecast)
print(f"Auto-ARIMA MAE: {mae:.2f}")
```

---

## 5. SARIMA (Seasonal ARIMA) 📆

```python
# SARIMA(p,d,q)(P,D,Q,s)
# s = seasonal period (12 for monthly data)

from statsmodels.tsa.statespace.sarimax import SARIMAX

model = SARIMAX(
    train['value'],
    order=(1, 1, 1),          # Non-seasonal
    seasonal_order=(1, 1, 1, 12)  # Seasonal (12 months)
)

model_fit = model.fit(disp=False)

# Forecast
forecast = model_fit.forecast(steps=len(test))

mae = mean_absolute_error(test['value'], forecast)
print(f"SARIMA MAE: {mae:.2f}")
```

---

## 6. Model Diagnostics 🔍

```python
# Residual analysis
residuals = model_fit.resid

# Plot residuals
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Residuals over time
axes[0, 0].plot(residuals)
axes[0, 0].set_title('Residuals')

# Histogram
axes[0, 1].hist(residuals, bins=30)
axes[0, 1].set_title('Histogram')

# ACF of residuals
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals, ax=axes[1, 0])

# Q-Q plot
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 1])

plt.tight_layout()
plt.show()

# Ljung-Box test (residuals should be white noise)
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=10)
print(lb_test)
```

---

## Quick Reference 📖

**Choosing ARIMA Parameters:**

**From ACF/PACF:**
- ACF cuts off at lag q → MA(q)
- PACF cuts off at lag p → AR(p)
- Both decay → ARMA(p,q)

**Grid Search:**
```python
import itertools

p = q = range(0, 3)
d = range(0, 2)
pdq = list(itertools.product(p, d, q))

best_aic = np.inf
best_params = None

for param in pdq:
    try:
        model = ARIMA(train['value'], order=param)
        results = model.fit()

        if results.aic < best_aic:
            best_aic = results.aic
            best_params = param
    except:
        continue

print(f"Best ARIMA{best_params}: AIC={best_aic:.2f}")
```

---

## Key Takeaways 💡

1. **ARIMA** = classical forecasting method
2. **Auto-ARIMA** finds best parameters automatically
3. **SARIMA** for seasonal data
4. **Check residuals** (should be white noise)
5. **AIC/BIC** for model selection
6. Works best for **stationary** series

---

**Next:** [Lesson 3 - Prophet & Modern Approaches →](Lesson%203%20-%20Prophet%20and%20Modern%20Approaches.md)
