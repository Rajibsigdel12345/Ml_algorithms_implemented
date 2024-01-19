# Machine Learning Algorithms Implementation

This repository contains Python implementations of following machine learning algorithms:

1. Linear Regression
2. Logistic Regression
3. Multiclass Logistic Regression using Softmax
4. Gaussian Discriminative Analysis

## 1. Linear Regression

### Mathematical Derivation:

Linear regression aims to model the relationship between a dependent variable and independent variables by fitting a linear equation. The hypothesis function is given by:

$\ h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n \$

To minimize the mean squared error, the parameters ($\theta\$) are found by solving the normal equation:

$\ \theta = (X^T X)^{-1} X^T y \$

### Usage:

To use the Linear Regression model:

1. **Instantiate the model:**

    $\ h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n \$

2. **Fit the model:**

    $\ \theta = (X^T X)^{-1} X^T y \$

3. **Make predictions:**

    $\ \text{Predictions} = X \theta \$

```python
from linear_regression import LinearRegression

# Instantiate the model
model = LinearRegression(lr=0.1, n_iters=1000)

# Fit the model
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
```


## 2. Logistic Regression

### Binary Classification using Sigmoid

Logistic regression models the probability of the positive class using the sigmoid function:

$\ h_\theta(x) = \frac{1}{1 + e^{-(\theta_0 + \theta_1 x_1 + \theta_2 x_2 + \ldots + \theta_n x_n)}} \$

#### Mathematical Derivation:

The hypothesis function $\( h_\theta(x) \) \$ is the sigmoid of a linear combination of input features, where $\( \theta \) \$ represents the model parameters.

The cost function for logistic regression is the negative log likelihood:

$\ (J(\theta) = -\frac{1}{m} \Sigma_{i=1}^{m} \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1 - y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right] \$

To optimize parameters $\( \theta \) \$, use gradient descent:

$\theta_j = \theta_j - \alpha \frac{1}{m} \Sigma_{i=1}^{m} \left( h_\theta(x^{(i)}) - y^{(i)} \right) x_j^{(i)} \$

where $\( \alpha \)\$ is the learning rate.


### Multiclass Classification using Softmax

For multiclass classification, logistic regression is extended using the softmax function. The hypothesis function for class $\(k\)\$ is given by:

$\ P(Y = k | X) = \frac{e^{\theta_k^T x}}{\Sigma_{j=1}^{K} e^{\theta_j^T x}} \$

#### Mathematical Derivation:

The softmax function assigns probabilities to each class, and the cost function is the cross-entropy loss:

$\ J(\theta) = -\frac{1}{m} \Sigma_{i=1}^{m} \Sigma_{k=1}^K \left[ y_k^{(i)} \log\left(\frac{e^{\theta_k^T x^{(i)}}}{\Sigma_{j=1}^{K} e^{\theta_j^T x^{(i)}}}\right) \right] \$

To optimize parameters $\( \theta \) \$, use gradient descent:

$\ \theta_{ij} = \theta_{ij} - \alpha \frac{1}{m} \Sigma_{i=1}^m \left( P(Y^{(i)} = j | X^{(i)}) - \mathbb{1}\{y^{(i)} = j\} \right) x_i \$

where $\( \alpha \)\$ is the learning rate, $\( K \)\$ is the number of classes, $\( \mathbb{1}\{\cdot\} \)\$ is the indicator function.

## 4. Gaussian Discriminative Analysis

Gaussian Discriminative Analysis assumes normal distribution within each class. The decision boundary is determined by setting the log-likelihood ratio equal to a threshold.

### Mathematical Derivation:

For each class $\ (k\) \$, GDA estimates the following parameters:

- Prior probability: $\ P(Y = k) = \frac{{\text{Number of samples in class } k}}{{\text{Total number of samples}}} \ \$
- Mean vector: $\( \mu_k = \frac{1}{{\text{Number of samples in class } k}} \Sigma_{i=1}^{m} \mathbb{1}\{y^{(i)} = k\} x^{(i)} \) \$
- Covariance matrix: $\( \Sigma_k = \frac{1}{{\text{Number of samples in class } k}} \Sigma_{i=1}^{m} \mathbb{1}\{y^{(i)} = k\} (x^{(i)} - \mu_k)(x^{(i)} - \mu_k)^T \) \$



### Usage:

To use the Gaussian Discriminative Analysis model:

1. **Instantiate the model:**

    $\ \delta_k(x) = \log(P(Y=k)) + \log(P(X=x | Y=k)) \$

2. **Fit the model:**

    - Prior probability: $\ P(Y = k) = \frac{{\text{Number of samples in class } k}}{{\text{Total number of samples}}} \$
    - Mean vector: $\ \mu_k = \frac{1}{{\text{Number of samples in class } k}} \Sigma_{i=1}^{m} \mathbb{1}\{y^{(i)} = k\} x^{(i)} \$
    - Covariance matrix: $\ \Sigma_k = \frac{1}{{\text{Number of samples in class } k}} \Sigma_{i=1}^{m} \mathbb{1}\{y^{(i)} = k\} (x^{(i)} - \mu_k)(x^{(i)} - \mu_k)^T \$

3. **Make predictions:**

    - For each $\(k\) \$, calculate $\ \delta_k(x) \$.
    - Assign $\(x\)\$ to the class $\(k\)\$ that maximizes $\ \delta_k(x) \$.

The discriminant function for class $\(k\)\$ is given by:

$\ \delta_k(x) = \log(P(Y=k)) + \log(P(X=x | Y=k)) \$

The decision boundary is determined by comparing the discriminant functions.



