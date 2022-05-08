import networkx as nx
import numpy as np
import random
from genBase import GraphGen

class Price_s(GraphGen):
    def __init__(self, m: int, c = 1, gamma = 1):
        self.m = m # the maximum out-degree of a node
        self.c = c  
        self.gamma = gamma 
        # the probability of attaching to a node with in-degree k is
        #  \frac{(k + c) ^ gamma * p_k}{\sum_k {(k + c) ^ gamma * p_k}}
        # where p_k denotes the fraction of nodes with in-degree k

    def getRandomSubset(sampleList, weightList, sze):
        targets = set()
        while len(targets) < sze:
            x = random.choices(sampleList, weights = weight)
            targets.add(x[0])
        return targets

    def __call__(self, n: int) -> nx.DiGraph:
        G = nx.DiGraph()
        if m < 1 or m >= n: 
            G.add_nodes_from(range(0, n))
            return G

        G.add_nodes_from(range(0, m))
        sampleList = list(range(0, m))
        weight = [pow(c, gamma)] * m

        for i in range(m, n):
            G.add_node(i)
            targets = getRandomSubset(sampleList, weight, m)
            for j in targets:
                degree = G.in_degree(j)
                wDelta = pow(degree + 1 + c, gamma) - pow(degree + c, gamma)
                if wDelta > 0:
                    sampleList.append(j)
                    weight.append(wDelta)
                G.add_edge(i, j)
            sampleList.append(i)
            weight.append(pow(c, gamma))

        return G