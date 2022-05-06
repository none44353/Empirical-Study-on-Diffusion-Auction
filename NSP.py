import networkx as nx
import numpy as np

# load the social network
G = nx.DiGraph()
E = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 5), (3, 6), (5, 7), (8, 9)]
G.add_edges_from(E)

# load bids of agents
bid = [0, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

seller = 0
maxBid, secPrice = 0, 0

for i in G.neighbors(seller):
    if bid[i] > maxBid:
        secPrice = maxBid
        maxBid = bid[i]
    elif bid[i] > secPrice:
        secPrice = bid[i]
        
SocialWelfare = maxBid
Revenue = secPrice

reachableNodes = list((lambda i: nx.has_path(G, seller, i), G.nodes))
Optimal = np.max([bid[i] for i in reachableNodes])
WorstCaseEfficiencyRatio = SocialWelfare / Optimal

print(Revenue)
print(SocialWelfare, WorstCaseEfficiencyRatio)