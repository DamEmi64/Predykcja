from numpy.lib import save
import sklearn.datasets
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import igraph as ig
import leidenalg
import os

from Data import dataClass

def add_vertex_if_not_exists(graph, vertex_name):
    if not has_node(graph,vertex_name):
        graph.add_vertex(vertex_name)

def has_node(graph, name):
    try:
        graph.vs.find(name=name)
    except:
        return False
    return True

def readData():
    directory = './data/dataset'
    g = ig.Graph();
    data = {}
    edges = []
    attributes = []
    for filename in os.listdir(directory):
        # len skasować, za długo to trwa
        if os.path.isfile(os.path.join(directory, filename)) and len(edges)< 150:
            file = open(os.path.join(directory, filename))
            lines = file.readlines()
            numbers = np.array(lines[1:]).astype(np.float32)
            add_vertex_if_not_exists(g,int(numbers[0]))
            add_vertex_if_not_exists(g,int(numbers[1]))
            edges.append([int(numbers[0]),int(numbers[1])])
            attributes.append(np.mean(numbers[3:]))
            
            if int(numbers[0]) in data.keys():
                data[int(numbers[0])].append(numbers[3:])
            else:
                data[int(numbers[0])] = [numbers[3:]]  
            
    g.add_edges(edges,{"mean":attributes})
    return g, data

def get_features(array):
    x_array= []
    y_array =[]
    for idx in range(10,len(array)):
        y_array.append(array[idx])
        x_array.append(array[idx-10:idx])
        
    return x_array, y_array


def get_train_test_data():
    
    g, node_data = readData()
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)
    print("Społeczności:")
    community_data = {}
    data = {}
    for community in partition:
        community_name = ",".join(str(x) for x in community)
        for node in community:
            if node in node_data.keys():
                x = []
                y = []
                if community_name in data.keys():
                    community_data[community_name].append(node_data[node])
                    
                    for node_arr in node_data[node]:
                        x_arr,y_arr = get_features(node_arr)
                        
                        for idx in (0,len(x_arr)-1):
                            x.append(x_arr[idx])
                            y.append(y_arr[idx])
                else:
                    community_data[community_name] = [node_data[node]]
                    
                    for node_arr in node_data[node]:
                        x_arr,y_arr = get_features(node_arr)
                        
                        for idx in (0,len(x_arr)-1):
                            x.append(x_arr[idx])
                            y.append(y_arr[idx])
                            
                dataCl = dataClass()                        
                dataCl.x = x
                dataCl.y = y
                data[community_name] = dataCl
        print(community)
    
    return data

data = get_train_test_data()

for key in data.keys():
    
    dataCl = data[key]
    
    X_train, X_test, y_train, y_test = train_test_split(dataCl.x, dataCl.y, train_size=0.7)
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    data_to_write = []
    for idx in range(0,len(X_test)):
         data_to_write.append([idx,y_test[idx]])

    pd.DataFrame(data_to_write).to_csv("./data/csv/"+key.replace(",","_")+"_test.csv")
    data_to_write = []
    for idx in range(0,len(X_test)):
         data_to_write.append([idx,y_pred[idx]])

    pd.DataFrame(data_to_write).to_csv("./data/csv/"+key.replace(",","_")+"_pred.csv")