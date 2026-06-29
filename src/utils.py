import pandas as pd
import networkx as nx
import gzip

import torch
from torch_geometric.data import HeteroData

import re

def create_graph(df):
    graph = nx.MultiDiGraph()
    node_num = 0
    edge_num = 0
    nodes = dict()
    edges = dict()

    for i, row in df.iterrows():
        if row['s'] not in nodes:
            nodes[row['s']] = node_num
            node_num += 1
        if row['o'] not in nodes:
            nodes[row['o']] = node_num
            node_num += 1
        if row['p'] not in edges:
            edges[row['p']] = edge_num
            edge_num += 1
        graph.add_edge(nodes[row['s']], nodes[row['o']], type = edges[row['p']])
        
    return graph

def load_ore_files(pathfilename):
    
    nt_file = open(pathfilename,'r',newline='\n')
    lines = nt_file.read().split('\r\n')

    entity_dict = dict({'s':[],'p':[],'o':[]})

    for line in lines[:-1]:
        split = line.split(' ')
        entity_dict['s'].append(re.findall(r'(?<=\().+',split[0])[0])
        entity_dict['p'].append(re.findall(r'.+?(?=\()',split[0])[0])
        entity_dict['o'].append(split[1].replace(')',''))

    df = pd.DataFrame(entity_dict)
    
    return df

def load_ore_graphs(path, train_file, test_file):
    print('Running...', train_file, test_file)

    df_train = load_ore_files(path+train_file)
    g_train = create_graph(df_train)
    
    df_train_filter_subclass = df_train[df_train['p'] == 'SubClassOf']
    g_train_filter_subclass = create_graph(df_train_filter_subclass)
    
    df_train_filter_assertion = df_train[df_train['p'] == 'ClassAssertion']
    g_train_filter_assertion = create_graph(df_train_filter_assertion)
    
    print(f'# Train - Triplets: {len(df_train)}, # Nodes: {g_train.number_of_nodes()}, # Edges: {g_train.number_of_edges()}')

    df_test = load_ore_files(path+test_file)
    g_test = create_graph(df_test)
    
    df_test_filter_subclass = df_test[df_test['p'] == 'SubClassOf']
    g_test_filter_subclass = create_graph(df_test_filter_subclass)
    
    df_test_filter_assertion = df_test[df_test['p'] == 'ClassAssertion']
    g_test_filter_assertion = create_graph(df_test_filter_assertion)
    
    print(f'# Test - Triplets: {len(df_test)}, # Nodes: {g_test.number_of_nodes()}, # Edges: {g_test.number_of_edges()}')
    
    print()
    
    return g_train, g_train_filter_subclass, g_train_filter_assertion, \
           g_test, g_test_filter_subclass, g_test_filter_assertion

def load_clg_files(pathfilename):
    
    nt_file = open(pathfilename,'r')
    lines = nt_file.readlines()
    lines = lines[0].split(' .')
    
    entity_dict = dict({'s':[],'p':[],'o':[]})
    
    for line in lines[:-1]:
        entities = line.split(' ')
        entity_dict['s'].append(entities[0])
        entity_dict['p'].append(entities[1])
        entity_dict['o'].append(entities[2])
        
    df = pd.DataFrame(entity_dict)

    return df

def load_clg_graphs(path, train_file, test_file):
    print('Running...', train_file, test_file)

    df_train = load_clg_files(path+train_file)
    g_train = create_graph(df_train)
    
    df_train_filter_subclass = df_train[df_train['p'] == '<http://www.w3.org/2000/01/rdf-schema#subClassOf>']
    g_train_filter_subclass = create_graph(df_train_filter_subclass)
    
    df_train_filter_assertion = df_train[df_train['p'] == '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>']
    g_train_filter_assertion = create_graph(df_train_filter_assertion)
    
    print(f'# Train - Triplets: {len(df_train)}, # Nodes: {g_train.number_of_nodes()}, # Edges: {g_train.number_of_edges()}')

    df_test = load_clg_files(path+test_file)
    g_test = create_graph(df_test)
    
    df_test_filter_subclass = df_test[df_test['p'] == '<http://www.w3.org/2000/01/rdf-schema#subClassOf>']
    g_test_filter_subclass = create_graph(df_test_filter_subclass)
    
    df_test_filter_assertion = df_test[df_test['p'] == '<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>']
    g_test_filter_assertion = create_graph(df_test_filter_assertion)
    
    print(f'# Test - Triplets: {len(df_test)}, # Nodes: {g_test.number_of_nodes()}, # Edges: {g_test.number_of_edges()}')
    
    print()
    
    return g_train, g_train_filter_subclass, g_train_filter_assertion, \
           g_test, g_test_filter_subclass, g_test_filter_assertion