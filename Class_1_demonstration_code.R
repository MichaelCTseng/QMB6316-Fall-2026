
# Simulating and Estimating a Simple Linear Regression Model

# Set population parameters
beta_0 <- 2
beta_1 <- 3
sigma  <- 2.5
n      <- 20

# Generate one random sample
#set.seed(123)

X <- runif(n, min = 0, max = 10)
#X <- rnorm(n, mean =2, sd = sigma)

epsilon <- rnorm(n, mean = 0, sd = sigma)
Y <- beta_0 + beta_1 * X + epsilon

# Estimate the regression by OLS
model <- lm(Y ~ X)
summary(model)

# Scatter plot
plot(X, Y,
     pch = 19, col = "gray40",
     main = "Population vs. Fitted Regression Line",
     xlab = "X", ylab = "Y")

# Population regression line (green)
abline(a = beta_0, b = beta_1,
       col = "green3", lwd = 2)

# Fitted regression line (red)
abline(model, col = "red", lwd = 2)

# Legend
legend("topleft",
       legend = c("Population regression line",
                  "Fitted regression line"),
       col = c("green3", "red"),
       lwd = 2, bty = "n")



#################################################



# Simulating and Estimating an Intercept-Only Linear Regression Model

# Set population parameters
beta_0 <- 2
sigma  <- 1
n      <- 10

# Generate one random sample
#set.seed(123)
epsilon <- rnorm(n, mean = 0, sd = sigma)
Y <- beta_0 + epsilon

# Estimate the intercept-only regression by OLS
model <- lm(Y ~ 1)
summary(model)

# Scatter plot: Y_i versus observation index i
plot(1:n, Y,
     pch = 19, col = "gray40",
     main = "Intercept-Only Regression",
     xlab = "Observation Index (i)", ylab = "Y")

# Population regression line (green)
abline(h = beta_0, col = "green3", lwd = 2)

# Fitted regression line (red)
abline(h = coef(model)[1], col = "red", lwd = 2)

# Legend
legend("topright",
       legend = c("Population regression line",
                  "Fitted regression line"),
       col = c("green3", "red"),
       lwd = 2, bty = "n")


mean(Y)

sqrt(var(Y)/n)




##################################################


# Simulating and Estimating a Linear Regression Model with a Dummy Variable

# Set population parameters
beta_0 <- 2
beta_1 <- 3
sigma  <- 2
n      <- 20

# Generate one random sample
# set.seed(123)
X <- rbinom(n, size = 1, prob = 0.5)  # Dummy variable: X = 0 or 1
epsilon <- rnorm(n, mean = 0, sd = sigma)
Y <- beta_0 + beta_1 * X + epsilon

# Estimate the regression by OLS
model <- lm(Y ~ X)
summary(model)

# Scatter plot
plot(X, Y,
     pch = 19, col = "gray40",
     main = "Regression with a Dummy Variable",
     xlab = "X (Dummy Variable)", ylab = "Y",
     xlim = c(-0.2, 1.2),
     xaxt = "n")

axis(1, at = c(0, 1), labels = c("0", "1"))

# Population regression line (green)
abline(a = beta_0, b = beta_1,
       col = "green3", lwd = 2)

# Fitted regression line (red)
abline(model, col = "red", lwd = 2)

# Legend
legend("topleft",
       legend = c("Population regression line",
                  "Fitted regression line"),
       col = c("green3", "red"),
       lwd = 2, bty = "n")



# Compute sample conditional means
Y_bar_0 <- mean(Y[X == 0])
Y_bar_1 <- mean(Y[X == 1])

Y_bar_0

Y_bar_1 - Y_bar_0


# Verify OLS estimates
coef(model)[1]  # Sample mean of Y given X = 0
coef(model)[2]  # Difference between the two sample means


#########################################################


# Simulating Bernoulli Data and Estimating an Intercept-Only Linear Regression

# Set population parameters
p <- 0.6
n <- 10

# Generate one Bernoulli sample
# set.seed(123)
Y <- rbinom(n, size = 1, prob = p)

# Estimate the intercept-only regression by OLS
model <- lm(Y ~ 1)
summary(model)

# Compute the sample mean
mean(Y)

# Scatter plot: Y_i versus observation index i
plot(1:n, Y,
     pch = 19, col = "gray40",
     main = "Intercept-Only Regression with Bernoulli Data",
     xlab = "Observation Index (i)", ylab = "Y",
     ylim = c(-0.1, 1.1))

# Population mean (green)
abline(h = p, col = "green3", lwd = 2)

# Fitted regression line (red)
abline(h = coef(model)[1], col = "red", lwd = 2)

# Legend
legend("right",
       legend = c("Population mean",
                  "Fitted regression line"),
       col = c("green3", "red"),
       lwd = 2, bty = "n")


# Standard error of the estimated intercept
SE <- summary(model)$coefficients[1, 2]
SE

# 95% confidence interval (using the t-distribution)
# confint(model, level = 0.95)


# 95% confidence interval (using the standard normal distribution)
p_hat <- coef(model)[1]
z <- qnorm(0.975)

CI <- c(p_hat - z * SE, p_hat + z * SE)
CI


# Compare with the standard error for a Bernoulli sample proportion
p_hat <- mean(Y)
SE_Bernoulli <- sqrt(p_hat * (1 - p_hat) / n)
SE_Bernoulli


###########################################################



# Monte Carlo Simulation: 95% Confidence Interval Coverage for a Bernoulli Model

# Set population parameters
p <- 0.4
n <- 10
M <- 100                     # Number of Monte Carlo samples
z <- qnorm(0.975)            # 95% standard normal critical value

set.seed(123)

# Store Monte Carlo results
p_hat <- numeric(M)
SE <- numeric(M)
CI_lower <- numeric(M)
CI_upper <- numeric(M)
coverage <- logical(M)

# Monte Carlo simulation
for (m in 1:M) {
  
  # Generate one Bernoulli sample
  Y <- rbinom(n, size = 1, prob = p)
  
  # Estimate intercept-only regression
  model <- lm(Y ~ 1)
  
  # Estimated intercept and standard error
  p_hat[m] <- coef(model)[1]
  SE[m] <- summary(model)$coefficients[1, 2]
  
  # 95% confidence interval using normal approximation
  CI_lower[m] <- p_hat[m] - z * SE[m]
  CI_upper[m] <- p_hat[m] + z * SE[m]
  
  # Check whether the confidence interval contains the true p
  coverage[m] <- (CI_lower[m] <= p) & (p <= CI_upper[m])
}

# Empirical coverage probability
mean(coverage)

# Compare nominal and empirical coverage probabilities
cat("Nominal coverage probability: 0.95\n")
cat("Empirical coverage probability:", mean(coverage), "\n")


##########################################################



# Bootstrap 95% Confidence Interval for an Intercept-Only Bernoulli Regression

# Set population parameters
p <- 0.4
n <- 10
B <- 1000                     # Number of bootstrap samples

# Generate one Bernoulli sample
set.seed(123)
Y <- rbinom(n, size = 1, prob = p)

# Estimate the intercept-only regression
model <- lm(Y ~ 1)
summary(model)

p_hat <- coef(model)[1]

# Bootstrap simulation
p_boot <- numeric(B)

for (b in 1:B) {
  
  # Resample with replacement from the original sample
  Y_boot <- sample(Y, size = n, replace = TRUE)
  
  # Estimate the intercept-only regression
  model_boot <- lm(Y_boot ~ 1)
  p_boot[b] <- coef(model_boot)[1]
}

# Bootstrap standard error
SE_boot <- sd(p_boot)
SE_boot

# 95% bootstrap confidence interval (percentile method)
CI_boot <- quantile(p_boot, probs = c(0.025, 0.975))
CI_boot

# Compare with the normal-approximation confidence interval
CI_normal <- p_hat + qnorm(c(0.025, 0.975)) *
  summary(model)$coefficients[1, 2]

CI_normal

# Check whether the bootstrap CI contains the true p
CI_boot[1] <= p && p <= CI_boot[2]


##################################################################################



# Monte Carlo Simulation: Bootstrap 95% CI Coverage for a Bernoulli Model

# Set population parameters
p <- 0.4
n <- 10
M <- 100                       # Number of Monte Carlo samples
B <- 1000                      # Number of bootstrap samples per MC sample

#set.seed(123)

# Store Monte Carlo results
p_hat <- numeric(M)
CI_lower <- numeric(M)
CI_upper <- numeric(M)
coverage <- logical(M)

# Monte Carlo simulation
for (m in 1:M) {
  
  # Generate one Bernoulli sample
  Y <- rbinom(n, size = 1, prob = p)
  
  # Estimate intercept-only regression
  model <- lm(Y ~ 1)
  p_hat[m] <- coef(model)[1]
  
  # Bootstrap simulation
  p_boot <- numeric(B)
  
  for (b in 1:B) {
    
    # Resample with replacement
    Y_boot <- sample(Y, size = n, replace = TRUE)
    
    # Estimate intercept-only regression on bootstrap sample
    model_boot <- lm(Y_boot ~ 1)
    p_boot[b] <- coef(model_boot)[1]
  }
  
  # 95% bootstrap confidence interval (percentile method)
  CI <- quantile(p_boot, probs = c(0.025, 0.975))
  
  CI_lower[m] <- CI[1]
  CI_upper[m] <- CI[2]
  
  # Check whether the bootstrap CI contains the true p
  coverage[m] <- (CI_lower[m] <= p) & (p <= CI_upper[m])
}

# Empirical coverage probability
mean(coverage)

# Compare nominal and empirical coverage probabilities
cat("Nominal coverage probability: 0.95\n")
cat("Empirical bootstrap coverage probability:", mean(coverage), "\n")
