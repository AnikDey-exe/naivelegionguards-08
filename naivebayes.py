import csv
import pandas as pd
import numpy as np
import plotly_express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from six import StringIO
from sklearn.preprocessing import StandardScaler 
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import seaborn as sns
import pydotplus
from IPython.display import Image
from sklearn import datasets

df = pd.read_csv("dincome.csv")
wine = datasets.load_wine()

print(wine)
x = wine["alcohol"].to_list()
y = wine["proline"].to_list()

# print(df.head())
# print(df.describe())

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
model = GaussianNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ",accuracy)

x_train_2, x_test_2, y_train_2, y_test_2 = train_test_split(x, y, test_size = 0.25, random_state = 42)

sc = StandardScaler()
x_train_2 = sc.fit_transform(x_train_2)
x_test_2 = sc.fit_transform(x_test_2)
model_2 = LogisticRegression(random_state = 0)
model_2.fit(x_train_2, y_train_2)

y_pred_2 = model_2.predict(x_test_2)
accuracy_2 = accuracy_score(y_test_2, y_pred_2)
print("Accuracy: ",accuracy_2)