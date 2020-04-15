import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
import pandas as pd

# Importing Data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Taking Care of Missing Data
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Encoding Independent Variables
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))

# Encoding Dependent Variable
le = LabelEncoder()
y = le.fit_transform(y)

# Feature Scaling
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting Data Into Training Set And Test Set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)