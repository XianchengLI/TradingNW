# -*-coding:utf-8-*-
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

def scv2nw_sole_product(df_year,SITC,exp=1,Cmap = None):
    # SITC: product SITC id
    # exp = 1, export_value; exp = 0, import_value
    # Cmap = map between country and Id
    df_tmp = df_year[df_year['SITC'] == SITC]
    if Cmap == None:
        list1 = set(df_tmp['ori'])
        Cmap = dict(zip(list1, range(1, len(list1) + 1)))
    df_tmp['ori'] = df_tmp['ori'].map(Cmap)
    df_tmp['des'] = df_tmp['des'].map(Cmap)
    if exp == 1:
        df_mat = df_tmp.pivot_table(['exp_v'], index='ori', columns='des')
    elif exp == 0:
        df_mat = df_tmp.pivot_table(['imp_v'], index='ori', columns='des')
    else:
        return "exp_type error"
    df_mat = df_mat.fillna(0)
    df_mat.columns = df_mat.index

    graph = nx.from_pandas_adjacency(df_mat)
    return graph,Cmap

def scv2nw_all_product(df_year,exp=1,Cmap = None):
    # SITC: product SITC id
    # exp = 1, export_value; exp = 0, import_value
    # Cmap = map between country and Id

    # method 0 : iterative
    # method 1 : eigenvalue
    # cal_type:
    # 1: power of exporter
    # 2: power of importer
    # 3: dependence of exporter
    # 4: dependence of importer
    df_tmp = df_year.copy()
    if Cmap == None:
        list1 = set(df_tmp['ori'])
        Cmap = dict(zip(list1, range(1, len(list1) + 1)))
    df_tmp['ori'] = df_tmp['ori'].map(Cmap)
    df_tmp['des'] = df_tmp['des'].map(Cmap)
    df_tmp = df_tmp.dropna()
    df_tmp['ori'] = df_tmp['ori'].map(lambda x: int(x))
    df_tmp['des'] = df_tmp['des'].map(lambda x: int(x))
    df_sum = df_tmp.groupby(['ori', 'des']).sum()
    # df_sum.loc[(0,slice(None)),:]
    # select in multi-index
    df_agg = df_tmp[['ori','exp_v','imp_v']].groupby('ori').sum()
    #df_agg.index.name = None
    # delete the name of index
    if exp == 1:
        df_mat = df_sum.pivot_table(['exp_v'], index='ori', columns='des')
    elif exp == 0:
        df_mat = df_sum.pivot_table(['imp_v'], index='ori', columns='des')
    else:
        return "exp_type error"
    df_mat = df_mat.fillna(0)
    df_mat.columns = df_mat.index

    graph = nx.from_pandas_adjacency(df_mat,create_using = nx.MultiDiGraph())
    return graph,Cmap,df_agg

def create_country_list():
    pre_path = 'D:\Data\Data from OEC\SITC_'
    year = 1962
    res = set()
    while year <= 2013:
        path = pre_path + str(year) + '.csv'
        df_year = pd.read_csv(path, header=None)
        df_year.columns = ['year', 'ori', 'des', 'SITC', 'exp_v', 'imp_v']
        s_tmp = set(df_year['ori'])
        res = res.union(s_tmp)
        year += 1
        print year
    li = list(res)
    li.sort()
    df = pd.DataFrame(li)
    df.to_csv('D:\Projects/tradingNW/node_l_coun.csv', header=None, index=None)

def create_edge_list(year,pre_path,node_l,exp=1,convert2int = 0):
    path = pre_path + str(year) + '.csv'
    df_year = pd.read_csv(path, header=None)
    df_year.columns = ['year', 'ori', 'des', 'SITC', 'exp_v', 'imp_v']
    df_tmp = df_year.copy()

    if convert2int == 1:
        d = node_l.to_dict()
        d = d[0]
        d = {v: k for k, v in d.items()}

        df_tmp['ori'] = df_tmp['ori'].map(d)
        df_tmp['des'] = df_tmp['des'].map(d)

    if exp == 1:
        df_tmp = df_tmp[[ u'ori', u'des', u'exp_v']]
        df_tmp = df_tmp[df_tmp['exp_v'] != 0]
    elif exp == 0:
        df_tmp = df_tmp[[u'ori', u'des', u'imp_v']]
        df_tmp = df_tmp[df_tmp['imp_v'] != 0]

    df_sum = df_tmp.groupby(['ori', 'des']).sum()
    df_sum = df_sum.reset_index()

    df_sum.to_csv('D:\Projects/tradingNW\edge_lists/'+str(year)+'_edge_l.csv', header=None, index=None)

def build_nw(edges_path,nodes_path):
    g = nx.read_edgelist(edges_path, delimiter=',', data=(('weight', float),))
    nx.write_gexf(g, 'D:\Projects/tradingNW\gephi_graphs/test_graph.gexf')



#def raw_Cmap(country_names):



#G = nx.Graph()
#Coun = pd.read_csv('D:\Projects/tradingNW\Data from OEC\country_names.csv')
#df_1962 = pd.read_csv('D:\Data\Data from OEC\SITC_1962.csv',header = None)
#4 exportvalue   5 importvalue
edges_path = 'D:\Projects/tradingNW\edge_lists/1962_edge_l.csv'
nodes_path = 'D:\Projects/tradingNW/node_l_coun.csv'


