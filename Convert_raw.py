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


G = nx.Graph()
#Coun = pd.read_csv('D:\Projects/tradingNW\Data from OEC\country_names.csv')
#df_1962 = pd.read_csv('D:\Data\Data from OEC\SITC_1962.csv',header = None)
#4 exportvalue   5 importvalue




