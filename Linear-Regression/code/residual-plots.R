
## Residual plots / diagnostics demo

## Florida oranges revisited
dat <-  read.csv("./dataset/florange.csv")
attach(dat)

plot(acres, boxes)
lm.1 <- lm(boxes ~ acres)
summary(lm.1)


# Residual plot vs fitted values
plot(lm.1$fitted.values, lm.1$residuals, xlab = "Fitted values", ylab = "Residuals") # cone-like: shows non-constant variance

# Residuals plot vs predictor (just one in this case)
plot(acres, lm.1$residuals, xlab = "Index", ylab = "Residuals") # also cone-like

# Residual plot: vs i (just to demo plot; no time/space ordering here)
plot(1:nrow(dat), lm.1$residuals, xlab = "Index", ylab="Residuals") # shows random scatter: no potential dependence over indices

# Histogram of residuals
hist(lm.1$residuals)

# QQ plot of residuals
qqnorm(lm.1$residuals)
qqline(lm.1$residuals, col = "blue", lwd = 2) # looks like the residuals are heavy tailed and don't follow a normal distribution. 


## Rocket data revisited
rocket <- read.csv("dataset/rocket.csv")
attach(rocket)

mr <- lm(thrust ~ nozzle + propratio)
summary(mr)

# Residual plot vs fitted values
plot(mr$fitted.values, mr$residuals, xlab = "Fitted values", ylab = "Residuals")

# Residual plot vs predictors
plot(nozzle, mr$residuals, xlab = "Nozzle (1 = large)", ylab = "Residuals")
plot(propratio, mr$residuals, xlab = "Propellant to fuel ratio", ylab = "Residuals")


# Histogram of residuals
hist(mr$residuals)


## QQ Plot of residuals
qqnorm(mr$residuals)
qqline(mr$residuals, col = "blue", lwd = 2) # data roughly follows the qqline. So no apparent violations


## Mystery dataset!
mystery <- read.csv("dataset/mystery.csv")
head(mystery)
attach(mystery)

pairs(mystery)
mm <- lm(Y ~ X1 + X2 + X3)
summary(mm)

plot(mm$fitted.values, mm$residuals, xlab = "Fitted values", ylab = "Residuals") # not a random scatter and follows a pattern.






