
# STAT 331 - Simple Linear Regression Demo

dat <- read.csv("dataset/florange.csv")

head(dat)
attach(dat)

# Scatterplot
plot(acres, boxes)

# Summary statistics calculation examples
r <- cor(acres, boxes)
xbar <- mean(acres)
ybar <- mean(boxes)
sd_x <- sd(acres) 
sd_y <- sd(boxes) 

# Manual Calculation Examples
Sxx <- sum( (acres - xbar)^2 )
Sxy <- sum( (acres - xbar) * (boxes - ybar))

# R's "lm" function fits linear models
lm.1 <- lm(boxes ~ acres)
summary(lm.1) # recall RSE = estimate of sigma-hat

# Fitted values
lm.1$fitted.values

# Residuals
lm.1$residuals

# Manual calculation of sigm^2 estimate
sum(lm.1$residuals^2) / 23
# or sigma estimate: rse
sqrt(sum(lm.1$residuals^2) / 23)

# t-distribution values
qt(0.975, 23)
(1 - pt(17.263, 23)) * 2


# Discussion:
# - is sigma the same for all values of y? It appears not to be --> this is testing the equal variance or homoestadastic assumption. This appears to be violated so we can consider taking the log transformation

# - Are the error terms plausibly independent? (e.g., does know one e_i help predict e_j for a different county?)




