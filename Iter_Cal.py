# -*-coding:utf-8-*-
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from Convert_raw import *
from math import sqrt

def MatLi2G(list):
    Matrix = np.mat(list)
    graph = nx.from_numpy_matrix(Matrix)

def Column_normalisation(m):
    #按列标准化
    i = 0
    lt = m.shape[1] #获取列数
    while i<lt :
        if sum(m[:, i]) != 0:
            m[:, i] = m[:, i] / sum(m[:, i])
        i += 1
    return m

def row_normalisation(m):
    #按行标准化
    i = 0
    lt = m.shape[0] #获取行数
    while i<lt :
        if m[i,:].sum() != 0:
            m[i,:] = m[i,:] / (m[i,:].sum())
        i += 1
    return m

def cal_dependenceOfExporter(Graph,method = 0):
    # method 0 : iterative
    # method 1 : eigenvalue
    w = nx.adjacency_matrix(Graph)
    m = w.todense()
    m = m.astype(np.float)
    m = row_normalisation(m)
    mm = m.dot(m.T)
    N = mm.shape[0]

    if method == 0:
        xx = np.ones([N, 1])
        xx = xx / np.linalg.svd(xx)[1][0]
        cvg = True
        ite = 0
        given_ite = 1000000
        while (cvg == True and ite < given_ite):
            xn = mm.dot(xx)
            xn = xn / np.linalg.svd(xn)[1][0]
            cvg = ((abs((xx.T).dot(xn)[0, 0] - 1.0)) > (10 ** (-10)))
            res = xn / sum(xn)
            xx = xn
            ite += 1
            # if ite % 100 == 0:
            #     print ite

    if method == 1:
        val, Vec = np.linalg.eig(mm)
        list_EV = val.tolist()[0]
        index = list_EV.index(max(list_EV))
        res = Vec[:,index]/sum(Vec[:,index])

        # f = find(v == max(v));
        # % eigen_dH = V(:, f)./ sum(V(:, f)); % % % dependence
        # of
        # H(eigen
        # method)

    return res

def cal_power_dependence(Graph,cal_type, method=0):
    # method 0 : iterative
    # method 1 : eigenvalue
    # cal_type:
    # 1: power of exporter
    # 2: power of importer
    # 3: dependence of exporter
    # 4: dependence of importer
    w = nx.adjacency_matrix(Graph)
    m = w.todense()
    m = m.astype(np.float)
    # print cal_type
    # print type(cal_type)
    if cal_type == 1:
        m = Column_normalisation(m)
        mm = m.dot(m.T)
    elif cal_type == 2:
        m = row_normalisation(m)
        mm = (m.T).dot(m)
    elif cal_type == 3:
        m = row_normalisation(m)
        mm = m.dot(m.T)
    elif cal_type == 4:
        m = Column_normalisation(m)
        mm = (m.T).dot(m)
    else :
        return 'cal_type_error'

    N = mm.shape[0]

    if method == 0:
        xx = np.ones([N, 1])
        xx = xx / np.linalg.svd(xx)[1][0]
        cvg = True
        ite = 0
        given_ite = 1000000
        while (cvg == True and ite < given_ite):
            xn = mm.dot(xx)
            xn = xn / np.linalg.svd(xn)[1][0]
            cvg = ((abs((xx.T).dot(xn)[0, 0] - 1.0)) > (10 ** (-20)))
            res = xn / sum(xn)
            xx = xn
            ite += 1
            # if ite % 100 == 0:
            #     print ite

    if method == 1:
        val, Vec = np.linalg.eig(mm)
        list_EV = val.tolist()
        index = list_EV.index(max(list_EV))
        res = Vec[:,index]/sum(Vec[:,index])

        # f = find(v == max(v));
        # % eigen_dH = V(:, f)./ sum(V(:, f)); % % % dependence
        # of
        # H(eigen
        # method)

    a = pd.DataFrame(res)
    a.index = Graph.node.keys()
    dic = a.to_dict()[0]

    return dic

def cal_set_all(Graph,method = 0):
    P_H = cal_power_dependence(Graph, 1, method)
    P_A = cal_power_dependence(Graph, 2, method)
    D_H = cal_power_dependence(Graph, 3, method)
    D_A = cal_power_dependence(Graph, 4, method)
    nx.set_node_attributes(Graph,P_H,name = 'P_H')
    nx.set_node_attributes(Graph, P_A, name='P_A')
    nx.set_node_attributes(Graph, D_H, name='D_H')
    nx.set_node_attributes(Graph, D_A, name='D_A')

def cal_year_power_dependence(year,exp,cal_type,dict_coun2ind,method):
    # cal_type:
    # 1: power of exporter
    # 2: power of importer
    # 3: dependence of exporter
    # 4: dependence of importer

    # exp 1 : use export data
    # exp 0 : use import data
    # normally, the two should get the same result

    # method 0 : iterative
    # method 1 : eigenvalue

    # dict_coun2ind: a dictionary to map countries and their ids

    pre_path = 'D:\Data\Data_from_OEC\SITC_'
    path = pre_path + str(year) + '.csv'
    df_year = pd.read_csv(path, header=None)
    df_year.columns = ['year', 'ori', 'des', 'SITC', 'exp_v', 'imp_v']
    #coun = pd.read_csv('D:\Data\Data from OEC\country_names.csv')
    # df_year['id_coun'] = df_year['ori'].map(dict_coun2ind)
    # df_year = df_year.dropna()
    # df_year['id_coun'] = df_year['id_coun'].map(lambda x: int(x))

    graph, Cmap,df_agg = scv2nw_all_product(df_year, exp, dict_coun2ind)
    # add isolated node
    for node_name in dict_coun2ind.values():
        if not (node_name in graph.nodes.keys()):
            graph.add_node(node_name)

    ########################## calculate the aggregated values for countries#####################
    # df_agg.index.name = None
    # df_agg_all = pd.merge(df_agg,coun[['cid']],left_index = True, right_index = True,how = 'right')
    # df_agg_all = df_agg_all.fillna(0)
    res = cal_power_dependence(graph, cal_type, method)
    return res

if __name__ == "__main__":
    # G = nx.Graph()
    # G.add_edge(1, 4)
    # G.add_edge(2, 4)
    # G.add_edge(3, 4)
    coun = pd.read_csv('D:\Projects/tradingNW\Countries_114.csv', sep=';')
    coun['index'] = coun.index
    #coun2ind = coun['id_3char']
    coun2ind = coun['cid']
    dict_coun2ind = coun2ind.to_dict()
    dict_coun2ind = {v: k for k, v in dict_coun2ind.items()}



    #nx.centrality.eigenvector_centrality(graph)

    #对单一产品计算
    # SITC = 10
    # graph,Cmap = scv2nw_sole_product(df_year, SITC, exp=1, Cmap=None)
    # res = cal_power_dependence(graph, 1, method=0)

