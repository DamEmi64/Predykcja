from numpy.lib import save
import sklearn.datasets
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
# Za³aduj dane
# Przygotuj dane do uczenia
# np. X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
def readData():
    # przerób
# zczytuje 2 i 3 jako wez³y 
# zczytuje dae do linijki

    datafile =  open("./data/dataset/1.txt")
    x = []
    y = []
    data = datafile.readlines()
    for idx in range(13,len(data)):            
        try:
            y.append(float(data[idx]))
            numbers = [float(data[1]),float(data[2])]
            for i in range(idx-10,idx):
                numbers.append(float(data[i]))
            x.append(numbers)
        except:
            pass
    return train_test_split(x, y, train_size=0.7)

X_train, X_test, y_train, y_test = readData();



# Utwórz model GNN
model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)

# Trenuj model
model.fit(X_train,y_train)

# Dokonaj predykcji
y_pred = model.predict(X_test)

data = []
for idx in range(0,len(X_test)):
     data.append([idx,y_test[idx]])

pd.DataFrame(data).to_csv("./data/csv/1_test.csv")
data = []
for idx in range(0,len(X_test)):
     data.append([idx,y_pred[idx]])

pd.DataFrame(data).to_csv("./data/csv/1_pred.csv")
input()