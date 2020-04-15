import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.svm import SVR
import pandas as pd

# Importing Data
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# # Taking Care of Missing Data
# imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
# X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# # Encoding Independent Variables
# ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
# X = np.array(ct.fit_transform(X))

# # Encoding Dependent Variable
# le = LabelEncoder()
# y = le.fit_transform(y)

# # Feature Scaling
# sc = StandardScaler()
# X = sc.fit_transform(X)

# # Splitting Data Into Training Set And Test Set
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Creating The Model
regressor = SVR(kernel = 'rbf', gamma = 0.1)
regressor.fit(X, y)

y_pred = regressor.predict([[6.5]])

# # Visualizing The Results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Title')
plt.xlabel('X Variable')
plt.ylabel('y Label')
plt.show()