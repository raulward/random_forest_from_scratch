from decision_tree.id3_decision_tree import DecisionTreeID3
import pandas as pd
import numpy as np
from sklearn import datasets

from sklearn.model_selection import train_test_split

np.random.seed(42)
n = 200  # número de instâncias

weather_options = ['Sunny', 'Overcast', 'Rain']
temp_options = ['Hot', 'Mild', 'Cool']
humidity_options = ['High', 'Normal']
wind_options = ['Weak', 'Strong']
play_options = ['Yes', 'No']

data = {
    'Weather': np.random.choice(weather_options, size=n),
    'Temperature': np.random.choice(temp_options, size=n),
    'Humidity': np.random.choice(humidity_options, size=n),
    'Wind': np.random.choice(wind_options, size=n),
    'Play': np.random.choice(play_options, size=n, p=[0.6, 0.4]) 
}

df = pd.DataFrame(data)

# print(df.head())


X = ['Weather', 'Temperature', 'Humidity', 'Wind']
y = 'Play'

X_train, X_test, y_train, y_test = train_test_split(df[X], df[y], test_size=0.2, random_state=42)

X_train.shape, X_test.shape, y_train.shape, y_test.shape


id3 = DecisionTreeID3()

id3.fit(X_train, y_train)

y_pred = id3.predict(X_test)

print(np.sum(y_pred == y_test) / len(y_test))

