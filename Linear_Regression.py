#importing necessary files
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#importing test and train file
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

#Linear Regression
from sklearn.linear_model import LinearRegression
lreg = LinearRegression()

#Splitting into Training and CV for Cross Validation
X = train.loc[:,['Outlet_Establishment_Year', 'Item_MRP']]
x_train, x_cv, y_train, y_cv = train_test_split(X, train.Item_Outlet_Sales)

#Training the Model
lreg.fit(x_train,y_train)

#Predicting on the Cross validation set
pred = lreg.predict(x_cv)

#Calculating the Mean Square Error
mse = np.mean((pred - y_cv)**2)
print('Mean Square Error is: ', mse)

#Calculation of coefficients
coeff = DataFrame(x_train.columns)
coeff['Coefficient Estimate'] = Series(lreg.coef_)
print(coeff)

x_plot = plt.scatter(pred, (pred - y_cv), c='b')
plt.hlines(y=0, xmin=-1000, xmax=5000)
plt.title('Residual Plot')
plt.show()
