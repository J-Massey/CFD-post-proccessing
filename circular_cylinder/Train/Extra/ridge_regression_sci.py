# we import necessary libraries and functions
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import numpy as np

data = np.load('../scaling_correction/data.npy')
X_train, X_test, y_train, y_test = train_test_split(data[:, 0:-1], data[:, -1], test_size=0.1, shuffle=False)

# we define an ordinary linear regression model (OLS)
linearModel=LinearRegression()
linearModel.fit(X_train, y_train)

# evaluating the model on training and testing sets
linear_model_train_r2=linearModel.score(X_train, y_train) # returns the r square value of the training set
linear_model_test_r2=linearModel.score(X_test, y_test)
print('Linear model training set r^2..:', linear_model_train_r2)
print('Lineer model testing r^2..:', linear_model_test_r2)

# here we define a Ridge regression model with lambda(alpha)=0.01
ridge=Ridge(alpha=0.01) # low alpha means low penalty
ridge.fit(X_train, y_train)
ridge_train_r2=ridge.score(X_train, y_train)
ridge_test_r2=ridge.score(X_test, y_test)
print('Ridge model (alpha=0.01) train r^2..:', ridge_train_r2)
print('Ridge model (alpha=0.01) test r^2..:', ridge_test_r2)

# we define a another Ridge regression model with lambda(alpha)=100
ridge2=Ridge(alpha=100) # when we increase alpha, model can not learn the data because of the low variance
ridge2.fit(X_train, y_train)
ridge2_train_r2=ridge2.score(X_train, y_train)
ridge2_test_r2=ridge2.score(X_test, y_test)
print('Ridge model (alpha=100) train r^2..:', ridge2_train_r2)
print('Ridge model (alpha=100) test r^2..:', ridge2_test_r2)

# visualize the values of the beta parameters
plt.figure(figsize=(8,6))
plt.plot(ridge.coef_, alpha=0.7, linestyle='none', marker='*', markersize=15, color='red', label=r'Ridge: $\lambda=0.01$')
plt.plot(ridge2.coef_, alpha=0.7, linestyle='none', marker='d', markersize=15, color='blue', label=r'Ridge: $\lambda=100$')
plt.plot(linearModel.coef_, alpha=0.7, linestyle='none', marker='v', markersize=15, color='orange', label=r'Linear Model')
plt.xlabel('Attributes', fontsize=16)
plt.ylabel('Attribute parameters', fontsize=16)
plt.legend(fontsize=16, loc=4)
plt.show()