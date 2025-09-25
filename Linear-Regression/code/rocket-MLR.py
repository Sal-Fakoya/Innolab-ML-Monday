import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# NASA rocket data example
# Title: High-Area Ratio Rocket Nozzle at High Combustion Chamber Pressure Experimental and Analytical Validation

# Read data
rocket = pd.read_csv("./dataset/rocket.csv")
print(rocket)

# Scatter plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(rocket['nozzle'], rocket['thrust'])
ax1.set_ylabel('Thrust')
ax1.set_xlabel('Nozzle size (1 = Large)')
ax1.set_title('Thrust vs Nozzle Size')

ax2.scatter(rocket['propratio'], rocket['thrust'])
ax2.set_ylabel('Thrust')
ax2.set_xlabel('Propellant to fuel ratio')
ax2.set_title('Thrust vs Propellant Ratio')

plt.tight_layout()
plt.show()

# Fit MLR using LinearRegression
X = rocket[['nozzle', 'propratio']]
y = rocket['thrust']

m1 = LinearRegression()
m1.fit(X, y)

print("Intercept:", m1.intercept_)
print("Coefficients:", m1.coef_)
print("R-squared:", m1.score(X, y))

# Using statsmodels for detailed summary (similar to R's summary())
X_sm = sm.add_constant(X)  # Adds intercept term
model_sm = sm.OLS(y, X_sm)
results = model_sm.fit()
print(results.summary())

# Manual beta estimates
X_matrix = np.column_stack([np.ones(len(rocket)), rocket['nozzle'], rocket['propratio']])
y_matrix = rocket['thrust'].values.reshape(-1, 1)
beta_hat = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ y_matrix
print("Manual beta estimates:")
print(beta_hat.flatten())

# Manual sigma estimate
mu_hat = X_matrix @ beta_hat  # fitted values
e = y_matrix - mu_hat  # residuals
sigma_hat = np.sqrt((e.T @ e) / (len(rocket) - 3))  # n - p - 1
print("Sigma hat:", sigma_hat[0, 0])

# Covariance matrix of beta_hat
print("Covariance matrix from statsmodels:")
print(results.cov_params())

# SEs of individual betas
print("SEs of individual betas from statsmodels:")
print(results.bse)

# Manual calculation of SEs of individual betas
se_beta = sigma_hat * np.sqrt(np.diag(np.linalg.inv(X_matrix.T @ X_matrix)))
print("Manual SEs of individual betas:")
print(se_beta.flatten())

# Estimate the mean response for units with small nozzle and propellant ratio 5.5. Include a 95% CI
new_data = pd.DataFrame({'nozzle': [0], 'propratio': [5.5]})

# Using scikit-learn for prediction
prediction = m1.predict(new_data)
print(f"Predicted thrust: {prediction[0]:.4f}")

# FIXED: Using statsmodels for confidence interval - ensure proper format
new_data_sm = sm.add_constant(new_data, has_constant='add')  # Explicitly add constant
print("New data for prediction:", new_data_sm.values)

# Method 1: Use the predict method directly
ci = results.get_prediction(new_data_sm).conf_int(alpha=0.05)
print("95% CI using statsmodels:")
print(f"Lower: {ci[0, 0]:.4f}, Upper: {ci[0, 1]:.4f}")

# Method 2: Alternative approach using predict method
pred_mean = results.predict(new_data_sm)
print(f"Predicted mean: {pred_mean.values[0]:.4f}")

# Manual calculation
x0 = np.array([[1, 0, 5.5]])
mu0_hat = (x0 @ beta_hat)[0, 0]
cov_matrix = np.linalg.inv(X_matrix.T @ X_matrix)
se_mu0 = sigma_hat[0, 0] * np.sqrt(x0 @ cov_matrix @ x0.T)[0, 0]
crit_val = stats.t.ppf(0.975, len(rocket) - 3)
ci_lo = mu0_hat - crit_val * se_mu0
ci_hi = mu0_hat + crit_val * se_mu0
print("Manual 95% CI:")
print(f"Estimate: {mu0_hat:.4f}, Lower: {ci_lo:.4f}, Upper: {ci_hi:.4f}")

# Predict the value of the response for a unit with a small nozzle and propellant ratio 5.5. Include a 95% PI
# Using statsmodels for prediction interval
pred_summary = results.get_prediction(new_data_sm).summary_frame(alpha=0.05)
print("95% Prediction Interval using statsmodels:")
print(f"Lower: {pred_summary['obs_ci_lower'].values[0]:.4f}, Upper: {pred_summary['obs_ci_upper'].values[0]:.4f}")

# Manual calculation for prediction interval
se_y0 = sigma_hat[0, 0] * np.sqrt(1 + (x0 @ cov_matrix @ x0.T)[0, 0])
pi_lo = mu0_hat - crit_val * se_y0
pi_hi = mu0_hat + crit_val * se_y0
print("Manual 95% Prediction Interval:")
print(f"Estimate: {mu0_hat:.4f}, Lower: {pi_lo:.4f}, Upper: {pi_hi:.4f}")

# Additional diagnostic information
print("\nAdditional Diagnostics:")
print(f"Residual standard error: {np.sqrt(results.mse_resid):.4f}")
print(f"Degrees of freedom: {results.df_resid}")
print(f"F-statistic: {results.fvalue:.4f}")
print(f"F-test p-value: {results.f_pvalue}")

# Verify the manual calculations match statsmodels
print("\nVerification - comparing manual vs statsmodels:")
print(f"Manual beta: {beta_hat.flatten()}")
print(f"Statsmodels beta: {results.params.values}")
print(f"Manual sigma: {sigma_hat[0, 0]:.6f}")
print(f"Statsmodels sigma: {np.sqrt(results.mse_resid):.6f}")