import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

# STAT 331 - Simple Linear Regression Demo

# Read data
dat = pd.read_csv("dataset/florange.csv")

print(dat.head())

# Scatterplot
plt.figure(figsize=(8, 6))
plt.scatter(dat['acres'], dat['boxes'])
plt.xlabel('acres')
plt.ylabel('boxes')
plt.title('Scatterplot of acres vs boxes')
plt.show()

# Summary statistics calculation examples
r = dat['acres'].corr(dat['boxes'])
xbar = dat['acres'].mean()
ybar = dat['boxes'].mean()
sd_x = dat['acres'].std()
sd_y = dat['boxes'].std()

print(f"Correlation (r): {r:.4f}")
print(f"Mean of acres (xbar): {xbar:.4f}")
print(f"Mean of boxes (ybar): {ybar:.4f}")
print(f"Standard deviation of acres: {sd_x:.4f}")
print(f"Standard deviation of boxes: {sd_y:.4f}")

# Manual Calculation Examples
Sxx = ((dat['acres'] - xbar) ** 2).sum()
Sxy = ((dat['acres'] - xbar) * (dat['boxes'] - ybar)).sum()

print(f"Sxx: {Sxx:.4f}")
print(f"Sxy: {Sxy:.4f}")

# Using scikit-learn for linear regression
X = dat[['acres']]  # Feature matrix
y = dat['boxes']    # Target variable

lm_1 = LinearRegression()
lm_1.fit(X, y)

print(f"Intercept: {lm_1.intercept_:.4f}")
print(f"Coefficient: {lm_1.coef_[0]:.4f}")

# Using statsmodels for more detailed summary (similar to R's summary())
X_sm = sm.add_constant(dat['acres'])  # Adds intercept term
model = sm.OLS(dat['boxes'], X_sm)
results = model.fit()
print(results.summary())

# Fitted values
fitted_values = lm_1.predict(X)
print("Fitted values:", fitted_values[:5])  # Show first 5

# Residuals
residuals = dat['boxes'] - fitted_values
print("Residuals:", residuals[:5])  # Show first 5

# Manual calculation of sigma^2 estimate
n = len(dat)
p = 1  # number of predictors
df = n - p - 1  # degrees of freedom = 23 (as in R code)
sigma_sq_hat = (residuals ** 2).sum() / df
rse = np.sqrt(sigma_sq_hat)

print(f"Sigma^2 estimate: {sigma_sq_hat:.4f}")
print(f"RSE: {rse:.4f}")

# t-distribution values
t_value = stats.t.ppf(0.975, 23)
print(f"t-value for 95% confidence with 23 df: {t_value:.4f}")

# For the p-value calculation (equivalent to (1 - pt(17.263, 23)) * 2)
t_stat = 17.263
p_value = (1 - stats.t.cdf(t_stat, 23)) * 2
print(f"p-value for t-statistic 17.263: {p_value:.6f}")

# Additional useful calculations
# R-squared
r_squared = lm_1.score(X, y)
print(f"R-squared: {r_squared:.4f}")

# Correlation matrix
corr_matrix = dat[['acres', 'boxes']].corr()
print("Correlation matrix:")
print(corr_matrix)






