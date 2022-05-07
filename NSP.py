from re import S
import networkx as nx
import numpy as np
import unittest
from mechanismBase import DiffusionAuction, getOptimal
import itertools

class NSP(DiffusionAuction):
    def __call__(self, G, seller):
        winner, maxBid, secPrice = seller, -1, -1
        bid = G.nodes.data("bid")
        for i in G.neighbors(seller):
            if bid[i] > maxBid:
                secPrice = maxBid
                maxBid = bid[i]
                winner = i
            elif bid[i] > secPrice:
                secPrice = bid[i]
        monetaryTransfer = dict(zip(G.nodes, itertools.repeat(0)))
        monetaryTransfer[seller] += secPrice
        monetaryTransfer[winner] -= secPrice
        return DiffusionAuction.MechanismResult(seller, winner, monetaryTransfer, G)

class TestNSP(unittest.TestCase):
    def test_NSP(self):
        mechanism = NSP()
        G = nx.DiGraph()
        E = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 5), (3, 6), (5, 7), (8, 9)]
        G.add_edges_from(E)
        # load bids of agents
        #bids = {0:0, 1:2, 2:3, 3:5, 4:7, 5:11, 6:13, 7:17, 8:19, 9:23, 10:29}
        bids = [0, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        seller = 0
        nx.set_node_attributes(G, dict(enumerate(bids)), "bid")

        result = mechanism(G, seller)    
        self.assertEqual(result.revenue(), 5)
        self.assertEqual(result.socialWelfare(), 7)
        self.assertAlmostEqual(result.efficiencyRatio(), 0.4117647058823529)


if __name__ == "__main__":
    unittest.main()