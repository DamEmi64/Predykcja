import igraph as ig
import leidenalg
import os
import numpy as np

# Tworzenie grafu
g = ig.Graph()

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
    # przer�b
# zczytuje 2 i 3 jako wez�y 
# zczytuje dae do linijki
    directory = './data/dataset'
    g = ig.Graph();
    edges = []
    attributes = []
    buffer = []
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)) and len(edges)< 150:
            file = open(os.path.join(directory, filename))
            lines = file.readlines()
            add_vertex_if_not_exists(g,lines[1])
            add_vertex_if_not_exists(g,lines[2])
            edges.append([lines[1],lines[2]])
            buffer.clear()
            
            for idx in range(3,len(lines)):            
                try:
                    buffer.append(float(lines[idx]))
                except:
                    pass
            attributes.append(np.mean(buffer))
    
    g.add_edges(edges,{"mean":attributes})
    return g

g = readData()


# Wykrywanie społeczności przy użyciu algorytmu Leiden
partition = leidenalg.find_partition(g, leidenalg.ModularityVertexPartition)

# Wyświetlanie wyników
print("Społeczności:")
for community in partition:
    print(community)