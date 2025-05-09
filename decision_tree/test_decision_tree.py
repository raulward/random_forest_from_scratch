from id3_decision_tree import DecisionTreeID3
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split


df = pd.read_csv('./data/test_data.csv')

# print(df.head())


X = ['Temperature', 'Humidity', 'Wind']
y = 'Play'

X_train, X_test, y_train, y_test = train_test_split(df[X], df[y], test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


id3 = DecisionTreeID3()

id3.fit(X_train, y_train)

y_pred = id3.predict(X_test)


# Acurácia -> 0.59 (aproximadamente)
print(np.sum(y_pred == y_test) / len(y_test))

