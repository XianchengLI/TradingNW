# -*-coding:utf-8-*-
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def MatLi2G(list):
    Matrix = np.mat(list)
    graph = nx.from_numpy_matrix(Matrix)

def plot_graph():
    nx.draw(G)
    plt.show()


G = nx.Graph()
G.add_edge(1, 4)
 # 添加边
G.add_edge(2, 4)
G.add_edge(3, 4)
w = nx.adjacency_matrix(G)
w = w.todense()
#  πij = wij / w.j
s = w.sum(axis = 0)
w = w.astype(np.float)
pi = w/s
pi2 = pi.dot(pi.T)
#pi2 = (pi.T).dot(pi)
x,y = np.linalg.eig(pi2)
# alphe

#


# print G.nodes()  # 输出所有的节点
# print G.edges()  # 输出所有的边
# print G.number_of_edges()  # 边的条数，只有一条边，就是（2，3）