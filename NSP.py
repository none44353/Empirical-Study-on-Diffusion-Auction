import networkx as nx
import numpy as np

def getOptimal(G, bid, seller):
    reachableNodes = list(filter(lambda i: nx.has_path(G, seller, i), G.nodes))
    Optimal = np.max([bid[i] for i in reachableNodes])
    return Optimal

def NSP(G, bid, seller):
    winner, maxBid, secPrice = 0, 0, 0
    for i in G.neighbors(seller):
        if bid[i] > maxBid:
            secPrice = maxBid
            maxBid = bid[i]
            winner = i
        elif bid[i] > secPrice:
            secPrice = bid[i]
    return winner, secPrice

def main():
    # load the social network
    G = nx.DiGraph()
    E = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 5), (3, 6), (5, 7), (8, 9)]
    G.add_edges_from(E)
    # load bids of agents
    #bid = {0:0, 1:2, 2:3, 3:5, 4:7, 5:11, 6:13, 7:17, 8:19, 9:23, 10:29}
    bid = [0, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    seller = 0

    winner, Revenue = NSP(G, bid, seller)    
    SocialWelfare = bid[winner]

    Optimal = getOptimal(G, bid, seller)
    WorstCaseEfficiencyRatio = SocialWelfare / Optimal

    print(Revenue)
    print(SocialWelfare, WorstCaseEfficiencyRatio)

if __name__ == "__main__":
    main()