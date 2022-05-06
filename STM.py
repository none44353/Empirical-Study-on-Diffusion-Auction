from audioop import reverse
import networkx as nx
import numpy as np
import IDM
import NSP

def getTopoGamma(G, seller):
    idom = nx.immediate_dominators(G, seller)
    Gamma = []
    for i in list(G.nodes):
        if (nx.has_path(G, seller, i) and idom[i] == seller):
            Gamma.append(i)
    return Gamma

def STM(G, bid, seller, Gamma):
    idom = nx.immediate_dominators(G, seller)
    reachableNodes = list(filter(lambda i: nx.has_path(G, seller, i), G.nodes))
    p, maxAlpha = IDM.getPrice(G, seller, bid, idom, reachableNodes)
    mxBidder, diffusionSeq = IDM.getDiffSeq(seller, reachableNodes, idom, bid)

    q = {}
    for ix in range(0, len(diffusionSeq)):
        i = diffusionSeq[ix]
        q[i] = p[i]

    for y in Gamma:
        if not (y in diffusionSeq):
            z = idom[y]
            while not (z in diffusionSeq):
                z = idom[z]
            q[z] = max(q[z], maxAlpha[y])
            
    winner, revenue = -1, q[seller]
    for ix in range(0, len(diffusionSeq)):
        i = diffusionSeq[ix]
        if (i == mxBidder):
            winner = i
        elif (bid[i] >= q[i]):
            winner = i
            break
        else:
            revenue = revenue + p[diffusionSeq[ix + 1]] - q[i]

    return winner, revenue

def main():
    # load the social network
    G = nx.DiGraph()
    E = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 5), (3, 6), (5, 7), (5, 8), (100, 101)]
    G.add_edges_from(E)
    # load bids of agents
    bid = [0, 2, 3, 5, 7, 13, 11, 17, 14, 23, 29]
    seller = 0
    Gamma = getTopoGamma(G, seller)
    winner, Revenue = STM(G, bid, seller, Gamma)
    SocialWelfare = bid[winner]

    Optimal = NSP.getOptimal(G, bid, seller)
    WorstCaseEfficiencyRatio = SocialWelfare / Optimal

    print(Revenue)
    print(SocialWelfare, WorstCaseEfficiencyRatio)

if __name__ == "__main__":
    main()