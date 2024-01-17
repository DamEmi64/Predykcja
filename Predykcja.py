from numpy.lib import save
from numpy.random import f
import scipy
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
from Classes import edgeDataClass

from Classes import dataClass

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
    data = []
    edges = []
    var_attribute = []
    skew_attribute = []
    kurtosis_attribute = []
    for filename in os.listdir(directory):
        # len skasować, za długo to trwa
        if os.path.isfile(os.path.join(directory, filename)):
            file = open(os.path.join(directory, filename))
            lines = file.readlines()
            numbers = np.array(lines[1:]).astype(np.float32)
            add_vertex_if_not_exists(g,int(numbers[0]))
            add_vertex_if_not_exists(g,int(numbers[1]))
            edges.append([int(numbers[0]),int(numbers[1])])
            var_attribute.append(np.var(numbers[3:]))
            skew_attribute.append(scipy.stats.skew(numbers[3:]))
            kurtosis_attribute.append(scipy.stats.kurtosis(numbers[3:]))
            dataCl = edgeDataClass(numbers[0],numbers[1],numbers[3:])
            
            data.append(dataCl)
            
    g.add_edges(edges,{"var":var_attribute, "skew":skew_attribute, "kurtosis": kurtosis_attribute})
    return g, data

def get_features(array):
    x_array= []
    y_array =[]
    for idx in range(10,len(array)):
        y_array.append(array[idx])
        x_array.append(array[idx-10:idx])
        
    return x_array, y_array


def get_communities():
    
    g, node_data = readData() 
    partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition, max_comm_size=int(len(g.vs)/3))
    communities = {}
    print("Społeczności:")
    for community in partition:
        print(community)
        community_name = ",".join(str(x) for x in community)
        communities[community_name] = []
        for dataCl in node_data:
            if dataCl.start_node in community and dataCl.end_node in community:
               communities[community_name].append(dataCl)
    
    return communities
'''
    data = {}
    for community in partition:
        community_name = ",".join(str(x) for x in community)
        x2 = []
        y2 = []
            model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
        for dataCl in node_data:
            if dataCl.start_node in community and dataCl.end_node in community:
                x_arr,y_arr = get_features(node_arr)
                X_train, X_test, y_train, y_test = train_test_split(x_arr, y_arr, train_size=0.7)
                

        for node in community:
            if node in node_data:

                if community_name in data.keys():
                    community_data[community_name].append(node_data[node])
                    
                    for node_arr in node_data[node]:
                        
                        
                        for idx in range(0,len(x_arr)-1):
                            x2.append(x_arr[idx])
                            y2.append(y_arr[idx])
                else:
                    community_data[community_name] = [node_data[node]]
                    
                    for node_arr in node_data[node]:
                        x_arr,y_arr = get_features(node_arr)
                        
                        for idx in range(0,len(x_arr)-1):
                            x2.append(x_arr[idx])
                            y2.append(y_arr[idx])
                            
        dataCl = dataClass()                        
        dataCl.x = x2
        dataCl.y = y2
        data[community_name] = dataCl
        print(community)
    
    return data
'''
communities = get_communities()

for community in communities.keys():
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
    test_data = {}
    train_data_x = []
    train_data_y = []
    edges = {}
    flaga = True
    for dataCl in communities[community]:
        x, y = get_features(dataCl.data)
        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.7)
        edge_name = str(dataCl.start_node) + "_" + str(dataCl.end_node)
        if flaga:
            flaga = False
            model2 = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000)
            model2.fit(X_train,y_train)
            y_pred = model2.predict(X_test)
            data_to_write = []
            for idx in range(0,len(y_pred)):
                data_to_write.append([idx,y_pred[idx]])
            pd.DataFrame(data_to_write).to_csv("./data/csv/"+edge_name+"_simple_pred.csv")  
            

        for idx in range(0,len(X_train)):
             train_data_x.append(X_train[idx])
             train_data_y.append(y_train[idx])

        if edge_name in edges.keys():
            data2Cl = edges[edge_name]
            for idx in range(0,len(X_test)):
                data2Cl.x.append(X_test[idx])
                data2Cl.y.append(y_test[idx])
            edges[edge_name] = data2Cl
        else:
            edges[edge_name] =  dataClass(X_test,y_test)
    model.fit(train_data_x[:250000],train_data_y[:250000])
    for edge in edges.keys():
        y_pred = model.predict(edges[edge].x)  
        data_to_write = []
        for idx in range(0,len(edges[edge].x)):
             data_to_write.append([idx,y_pred[idx]])
        pd.DataFrame(data_to_write).to_csv("./data/csv/"+edge+"_pred.csv")
        data_to_write = []
        for idx in range(0,len(edges[edge].x)):
             data_to_write.append([idx,edges[edge].y[idx]])
        pd.DataFrame(data_to_write).to_csv("./data/csv/"+edge+"_test.csv")

