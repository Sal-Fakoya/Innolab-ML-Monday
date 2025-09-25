# NASA rocket data example

# Title: High-Area Ration Rocket Nozzle at High Combustion Chamber Pressure Experimental and Analytical Validation

rocket <- read.csv(file="./dataset/rocket.csv")
rocket
attach(rocket)

# Scatter plots
par(mfrow = c(1, 2))
plot(nozzle, thrust, ylab = "Thrust", xlab = "Nozzle size (1 = Large)")
plot(propratio, thrust, ylab = "Thrust", xlab = "Propellant to fuel ratio")

# Fit MLR using lm: lm(y ~ x1 + x2)
m1 <- lm(thrust ~ nozzle + propratio)
summary(m1)

# Manual beta estimates
X <- cbind(rep(1, 12), nozzle, propratio)
y <- matrix(thrust, ncol = 1)
beta_hat <- solve(t(X) %*% X) %*% t(X) %*% y 
beta_hat

# Manual sigma estimate
mu_hat <- X %*% beta_hat # fitted values
e <- y - mu_hat # residuals
sigma_hat <- sqrt(t(e) %*% e / 9) # note: n - p - 1 = 9
sigma_hat
sigma_hat <- sqrt(sum(e^2) / 9) # equivalent
sigma_hat


# Covariance matrix of beta_hat
vcov(m1) # diagonals are estimated variances for corresponding betas
sqrt(diag(vcov(m1))) # SEs of individual betas

# Manual calculation of SEs of individual betas
se_beta <- sigma_hat * sqrt(diag(solve(t(X) %*% X)))
se_beta

# Estimate the mean response for units with small nozzle and propellant ratio 5.5. Include a 95% CI
predict(object = m1,
        newdata = data.frame(nozzle = 0, 
                              propratio = 5.5),
        interval = "confidence", level = 0.95)


# Manual calculation
x0 <- matrix(c(1, 0, 5.5), nrow = 1)
mu0_hat <- x0 %*% beta_hat
se_mu0 <- sigma_hat * sqrt(x0 %*% solve(t(X) %*% X) %*% t(x0))
crit_val <- qt(0.975, 9)
ci_lo <- mu_0hat - crit_val * se_mu0
ci_hi <- mu_0hat + crit_val * se_mu0
c(mu_0hat, ci_lo, ci_hi)


# Predict the value of the response for a unit with a small nozzle and propellant ratio 5.5. Include a 95% PI
predict(object = m1, 
        newdata = data.frame(nozzle = 0, 
                             propratio = 5.5),
        interval = "prediction", level = 0.95)

# Manual calculation
x0 <- matrix(c(1, 0, 5.5), nrow = 1)
mu0_hat <- x0 %*% beta_hat
se_y0 <- sigma_hat * sqrt(1 + x0 %*% solve(t(X) %*% X) %*% t(x0))
crit_val <- qt(0.975, 9)
pi_lo <- mu_0hat - crit_val * se_y0
pi_hi <- mu_0hat + crit_val * se_y0
c(mu_0hat, pi_lo, pi_hi)

