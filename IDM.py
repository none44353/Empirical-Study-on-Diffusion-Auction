from audioop import reverse
import networkx as nx
import numpy as np

def getPrice(G, seller, bid, idom, H):
    # H is the set of reachable vertexes from seller
    dist = nx.single_source_shortest_path_length(G, seller)
    print(idom[0], idom[1], idom[2], idom[3], idom[4], idom[5], idom[6])
    domTree = nx.DiGraph()
    for i in H:
        if i != seller: domTree.add_edge(idom[i], i)

    H.sort(key = (lambda i: dist[i]))
    #the maximum bid of i's subtree in domTree Graph, denoted as maxbid
    maxbid = {}
    for i in reversed(H):
        maxbid[i] = np.max([maxbid[j] for j in domTree.successors(i)] + [bid[i]])
    price = {seller: 0}
    for i in H:
        adj = list(domTree.successors(i))
        val = [maxbid[j] for j in adj]
        preMaxVal = [0] * (len(adj) + 1)
        for jx in range(0, len(adj)):
            preMaxVal[jx + 1] = max(preMaxVal[jx], val[jx])
        sufMaxVal = [0] * (len(adj) + 1)
        for jx in range(len(adj) - 1, -1, -1):
            sufMaxVal[jx - 1] = max(sufMaxVal[jx], val[jx])
        for jx in range(0, len(adj)):
            price[adj[jx]] = max(preMaxVal[jx], sufMaxVal[jx])

    return price

def getDiffSeq(seller, reachableNodes, idom):
    # get diffusion sequence C_{x^*} = {c_0 = s, c_1, c_2, ..., c_l = x^*}
    mxBidder = seller
    for i in reachableNodes:
        if bid[i] > bid[mxBidder]: mxBidder = i
    diffusionSeq, x = [], mxBidder
    while x != seller:
        diffusionSeq.append(x)
        x = idom[x]
    diffusionSeq.append(seller)
    diffusionSeq.reverse()
    return mxBidder, diffusionSeq

# load the social network
G = nx.DiGraph()
E = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 5), (3, 6), (5, 7), (8, 9)]
G.add_edges_from(E)
# load bids of agents
bid = [0, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29]

seller = 0
idom = nx.immediate_dominators(G, seller)
reachableNodes = list(filter(lambda i: nx.has_path(G, seller, i), G.nodes))
price = getPrice(G, seller, bid, idom, reachableNodes)
mxBidder, diffusionSeq = getDiffSeq(seller, reachableNodes, idom)

winner = -1
for ix in range(0, len(diffusionSeq)):
    i = diffusionSeq[ix]
    if (i == mxBidder):
        winner = i
    elif (price[diffusionSeq[ix + 1]] == bid[i]):
        winner = i

Revenue = price[diffusionSeq[1]]
SocialWelfare = bid[winner]

Optimal = np.max([bid[i] for i in reachableNodes])
WorstCaseEfficiencyRatio = SocialWelfare / Optimal

print(Revenue)
print(SocialWelfare, WorstCaseEfficiencyRatio)