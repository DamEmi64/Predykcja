import sklearn.datasets
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
# Za³aduj dane
# Przygotuj dane do uczenia
# np. X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def readData():
    data = sklearn.datasets.load_files('./data/dataset')
    x = []
    y = data.target
    for item in data.data:
        lines = item.splitlines()
        numbers = []
        for line in lines:
                try:
                    numbers.append(float(line))
                except :
                    pass
        x.append(numbers)

    return train_test_split(x, y, train_size=0.7)

X_train, X_test, y_train, y_test = readData();

data = []
for index in range(len(X_train)):
     data.append([X_train[index],y_train[index]])

# Przeskaluj dane
scaler = StandardScaler()
scaler.fit(data)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# Utwórz model GNN
model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000)

# Trenuj model
model.fit(X_train, y_train)

# Dokonaj predykcji
y_pred = model.predict(X_test)

# Ocena modelu
accuracy = accuracy_score(y_test, y_pred)
print("dokladnosc modelu" + accuracy)