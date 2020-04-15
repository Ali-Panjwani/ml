import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import pandas as pd

# Importing Data
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, 0:-1].values
y = dataset.iloc[:, 4:].values

# # Taking Care of Missing Data
# imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
# X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Encoding Independent Variables
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [3])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
X = X[:, 1:]

# # Encoding Dependent Variable
# le = LabelEncoder()
# y = le.fit_transform(y)

# # Feature Scaling
# sc_X = StandardScaler()
# sc_y = StandardScaler()
# X = sc_X.fit_transform(X)
# y = sc_y.fit_transform(y)

# # # Splitting Data Into Training Set And Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# # Creating The Model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
# y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

X= np.append(arr = np.ones((50,1)).astype(int), values = X, axis =1)
X_opt = X[:,[0,1,2,3,4,5]]
X_opt = np.array(X_opt, dtype=float)
regressor= sm.OLS(y, X_opt).fit()
print(regressor.summary())

X_opt = X[:,[0,1,3,4,5]]
X_opt = np.array(X_opt, dtype=float)
regressor= sm.OLS(y, X_opt).fit()

X_opt = X[:,[0,3,4,5]]
X_opt = np.array(X_opt, dtype=float)
regressor= sm.OLS(y, X_opt).fit()
print(regressor.summary())

X_opt = X[:,[0,3,5]]
X_opt = np.array(X_opt, dtype=float)
regressor= sm.OLS(y, X_opt).fit()
print(regressor.summary())

X_opt = X[:,[0,3]]
X_opt = np.array(X_opt, dtype=float)
regressor= sm.OLS(y, X_opt).fit()
print(regressor.summary())


