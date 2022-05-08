import networkx as nx
import numpy as np
import random

def getRandomSubset(sampleList, weightList, sze):
    targets = set()
    while len(targets) < sze:
        x = random.choices(sampleList, weights = weight)
        targets.add(x[0])
    return targets

c, gamma, m = 1, 1, 4
n = 10
if m < 1 or m >= n: 
    raise nx.NetworkXError("Price network must have m>=1 and m<n, m=%d,n=%d"%(m,n))

G = nx.DiGraph()
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

print(G)
print(list(G.edges()))
