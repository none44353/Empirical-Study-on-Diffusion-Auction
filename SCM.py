from audioop import reverse
from random import randint
import networkx as nx
import numpy as np
import IDM
import NSP
import STM

def getSybilClusterIndex(G, seller, Gamma):
    dist = nx.single_source_shortest_path_length(G, seller)
    reachableNodes = list(filter(lambda i: nx.has_path(G, seller, i), G.nodes))
    reachableNodes.sort(key = (lambda i: dist[i]))
    idom = nx.immediate_dominators(G, seller)
    clusterIndex = {}
    for i in reachableNodes:
        if i in Gamma:
            clusterIndex[i] = i
        else:
            clusterIndex[i] = clusterIndex[idom[i]]
    return clusterIndex

def getRSPTree(G, source): # get Random Shortest Path Tree 
    dist = nx.single_source_shortest_path_length(G, source)
    reachableNodes = list(filter(lambda i: nx.has_path(G, source, i), G.nodes))
    reachableNodes.sort(key = (lambda i: dist[i]))
    Tree = nx.DiGraph()
    for i in reachableNodes:
        if i != source:
            fromList = []
            for j in G.predecessors(i):
                if dist[j] + 1 == dist[i]: fromList.append(j)
            Tree.add_edge(fromList[randint(0, len(fromList) - 1)], i)
    return Tree

def SCM(G, bid, seller):
    Gamma = STM.getTopoGamma(G, seller)
    clusterIndex = getSybilClusterIndex(G, seller, Gamma)
    clustersGraph = nx.DiGraph()
    for i,j in G.edges():
        if nx.has_path(G, seller, i) and clusterIndex[i] != clusterIndex[j]:
            clustersGraph.add_edge(clusterIndex[i], clusterIndex[j])
    
    clustersTree = getRSPTree(clustersGraph, seller)
    subG = G.copy()
    for i,j in G.edges():
        if nx.has_path(G, seller, i) and clusterIndex[i] != clusterIndex[j]:
            if not clustersTree.has_edge(clusterIndex[i], clusterIndex[j]): 
                subG.remove_edge(i, j)
    return STM.STM(subG, bid, seller, Gamma)

def main():
    # load the social network
    G = nx.DiGraph()
    E = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (3, 6), (4, 7), (5, 7), (6, 7), (3, 8), (100, 101)]
    G.add_edges_from(E)
    # load bids of agents
    bid = [0, 2, 3, 5, 7, 13, 11, 17, 19, 23, 29]
    seller = 0
    winner, Revenue = SCM(G, bid, seller)
    SocialWelfare = bid[winner]

    Optimal = NSP.getOptimal(G, bid, seller)
    WorstCaseEfficiencyRatio = SocialWelfare / Optimal

    print(Revenue)
    print(SocialWelfare, WorstCaseEfficiencyRatio)

if __name__ == "__main__":
    main()