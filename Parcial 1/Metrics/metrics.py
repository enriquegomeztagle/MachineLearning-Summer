"""
# Evaluation Metrics

1. **MSE (Mean Squared Error)**: Measures the average of the squares of the errors, giving more weight to larger errors.
2. **RMSE (Root Mean Squared Error)**: The square root of MSE, providing an error metric in the same units as the target variable.
3. **RSS (Residual Sum of Squares)**: The sum of the squared differences between the observed and predicted values.
4. **SSR (Regression Sum of Squares)**: The sum of the squared differences between the predicted values and the mean of the observed values.
5. **TSS (Total Sum of Squares)**: The sum of the squared differences between the observed values and the mean of the observed values.
6. **R² (R-squared)**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variable(s).
7. **MAPE (Mean Absolute Percentage Error)**: The mean absolute percentage difference between the observed and predicted values, expressed as a percentage.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Generate sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Calculate metrics
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
rss = np.sum((y - y_pred) ** 2)
tss = np.sum((y - np.mean(y)) ** 2)
ssr = tss - rss
r2 = r2_score(y, y_pred)
mape = np.mean(np.abs((y - y_pred) / y)) * 100

# Prepare results in a dataframe
metrics = {
    'Metric': ['MSE', 'RMSE', 'RSS', 'SSR', 'TSS', 'R2', 'MAPE'],
    'Value': [mse, rmse, rss, ssr, tss, r2, mape]
}
metrics_df = pd.DataFrame(metrics)

# Display the DataFrame
print(metrics_df)