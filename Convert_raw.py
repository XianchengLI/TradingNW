# -*-coding:utf-8-*-
import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd

G = nx.Graph()
Coun = pd.read_csv('D:\Projects/tradingNW\Data from OEC\country_names.csv')
df_1962 = pd.read_csv('D:\Data\Data from OEC\SITC_1962.csv',header = None)
#4 exportvalue   5 importvalue
