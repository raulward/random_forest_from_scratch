from decision_tree.id3_decision_tree import DecisionTreeID3
import pandas as pd
import numpy as np
from random_forest.random_forest import RandomForest


from sklearn.model_selection import train_test_split

df = pd.read_csv('./data/test_data.csv')

# print(df.head())


X = ['Outlook', 'Temperature', 'Humidity', 'Wind']
y = 'Play'

X_train, X_test, y_train, y_test = train_test_split(df[X], df[y], test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

rf = RandomForest(n_trees=30)

rf.fit(X_train, y_train)

preds = rf.predict(X_test)

# Acurácia -> 0.69 (aproximadamente)
print(np.sum(preds == y_test) / len(y_test))
