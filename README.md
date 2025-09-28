# Innolab-ML-Monday
SLR and MLR code and dataset

Due to limited time, we will focus on Steps 4, 6, 7

### Summary of Steps for Linear Regression

1. Exploratory Data Analysis (EDA)
    - Distribution checks
    - Linearity assessment
    - Outlier detection
2. Variable Transformation
    - Response variable transformations
    - Predictor transformations
    - Interaction terms
3. Multicollinearity Check
    - VIF analysis
4. Model Selection Framework
    - Choose criteria (Adj R², AIC, BIC, MSPE)
    - Selection strategy (Brute force, Stepwise, etc.)
    - Significance Testing (F-test for nested model comparison)
5. Residuals Diagnostics: Critical to do for CI/PI validity due to normality assumption of errors around the predicted line
    - Normality of residuals (Q-Q plot)
    - Constant variance (Residuals vs Fitted)
    - Independence (Residuals vs Order)
    - Influential points (Leverage, Cook's Distance)
6. Final Model Validation & Interpretation
    - Out-of-sample testing (test dataset)
    - Cross-validation
    - Interpret Coefficients with Confidence Intervals (CIs)
7. Prediction
    - Make predictions for new observations
    - Report Prediction Intervals (PIs)

KEY NOTES:

- F-tests are used during model selection to compare nested models
- Residual diagnostics (Step 5) must pass before CIs and PIs can be trusted
- CIs quantify uncertainty in coefficient estimates
- PIs quantify uncertainty in individual predictions
