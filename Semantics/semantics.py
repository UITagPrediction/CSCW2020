# -*- coding: utf-8 -*-
"""
Created on Wed June 10 08:51:32 2019

@author: Sidong Feng
"""
import argparse
import pandas as pd
import networkx as nx
from apyori import apriori
import nltk 
from nltk.corpus import wordnet 

parser = argparse.ArgumentParser(description='UI Tag Semantics')
parser.add_argument('--support', type=float, default='0.003',
                    help='minimum support threshold (default: 0.003)')
parser.add_argument('--confidence', type=float, default='0.005',
                    help='minimum confidence threshold (default: 0.005)')
parser.add_argument('--lift', type=int, default='2',
                    help='minimum lift threshold (default: 2)')
parser.add_argument('--length', type=int, default='2',
                    help='minimum length threshold (default: 2)')
args = parser.parse_args()

# load dataset
def loadDataSet(FILE="../RecoverTags/Data/Metadata.csv"):
    df = pd.read_csv(FILE,encoding='ISO-8859-15', header = None, low_memory = False)
    df = df[[5]]
    df = df.dropna(axis=0,how='any')
    dataSet = []
    for _,row in df.iterrows():
        tags = row[5].strip().split("   ")
        tags = [x for x in tags if x not in ["ui", "user interface", "user_interface", "userinterface"]]
        dataSet.append(tags)
    return dataSet

# Create association rules based on apriori algorithm
def create_association(dataSet, min_support=args.support, min_confidence=args.confidence, min_lift=args.lift, min_length=args.length):
    associations = apriori(dataSet, min_support=min_support, min_confidence=min_confidence, min_lift=min_lift, min_length=min_length)
    associations = list(associations)
    return associations
 
# Create undirected graph based on the association rules
def rulesToGraph(associations):
    # construct nodes and edges
    nodes = []
    edges = []
    for association in associations:
        src, des, _, _ = association[2][0]
        nodes += list(src) + list(des)
        edges += [(x,y) for x in src for y in des]
    nodes = list(set(nodes))
    edges = list(set(edges))
    # create undirected graph
    G = nx.Graph()                                                       
    G.add_nodes_from(nodes)                          
    G.add_edges_from(edges)
    print(nx.info(G))
    return nodes, edges, G

# Create vocabulary based on the nodes
def create_vocabulary(nodes):
    vocabulary = {}
    for word in nodes:
        vocabulary[nodes] = [] 
        # find abbreviation and synonyms
        for syn in wordnet.synsets(word): 
            for l in syn.lemmas(): 
                vocabulary[nodes].append(l.name()) 
    return vocabulary

def main():
    """ load dataset """
    dataSet = loadDataSet()
    """ find association rules """
    associations = apriori(dataSet)
    """ create undirected graph """
    nodes, edges, G = rulesToGraph(associations)
    nx.write_gexf(G, "association.gexf")
    """ create vocabulary """
    vocabulary = create_vocabulary(nodes)

if __name__ == "__main__":
    print('-'*10)
    print(args)
    print('-'*10)
    main()
