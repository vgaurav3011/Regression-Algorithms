#Importing Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from pandas import Series, DataFrame
from sklearn.model_selection import train_test_split

#Importing the Training and Test Files
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')

#Splitting into Training and CV for Cross Validation
X = train.loc[:,['Outlet_Establishment_Year', 'Item_MRP']]
x_train, x_cv, y_train, y_cv = train_test_split(X, train.Item_Outlet_Sales)

#Lasso Regression
lassoReg = Lasso(alpha=0.5, normalize=True)
lassoReg.fit(x_train,y_train)
pred = lassoReg.predict(x_cv)

#Calculating the mean squared error
mse = np.mean((pred - y_cv)**2)
print('Mean Squared Error:',mse)
print('Score:',lassoReg.score(x_cv,y_cv))

#Calculation of coefficients
coeff = DataFrame(x_train.columns)
coeff['Coefficient Estimate'] = Series(lassoReg.coef_)
print(coeff)

#Plotting Analysis through a Residual Plot
x_plot = plt.scatter(pred, (pred - y_cv), c='b')
plt.hlines(y=0, xmin=-1000, xmax=5000)
plt.title('Residual Plot')
plt.show()

#Magnitude of Coefficents
predictors = x_train.columns
coef = Series(lassoReg.coef_,predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients')
plt.show()
