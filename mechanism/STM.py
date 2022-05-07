import networkx as nx
from IDM import IDM
from mechanismBase import DiffusionAuction
import unittest

class STM(DiffusionAuction):

    @staticmethod
    def getTopoGamma(G, seller):
        idom = nx.immediate_dominators(G, seller)
        Gamma = []
        for i in list(G.nodes):
            if (nx.has_path(G, seller, i) and idom[i] == seller):
                Gamma.append(i)
        return Gamma

    def __call__(self, G, seller, Gamma=None):
        bid = G.nodes.data("bid")
        if Gamma == None:
            Gamma = STM.getTopoGamma(G, seller)

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
                
        winner = -1
        monetaryTransfer = dict(zip(G.nodes, [0] * len(G.nodes)))
        for ix, i in enumerate(diffusionSeq):
            if (i == mxBidder) or (bid[i] >= q[i]):
                winner = i
                monetaryTransfer[i] = -p[i]
                monetaryTransfer[seller] += p[i]
                break
            else:
                monetaryTransfer[i] = q[i] - p[i]
                monetaryTransfer[seller] -= q[i] - p[i]

        return DiffusionAuction.MechanismResult(seller, winner, monetaryTransfer, G)

class TestSTM(unittest.TestCase):
    def testSTM_hand(self):
        stm = STM()
        G = nx.DiGraph()
        E = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 5), (3, 6), (5, 7), (5, 8), (100, 101)]
        G.add_edges_from(E)
        bid = [0, 2, 3, 5, 7, 13, 11, 17, 14, 23, 29]
        nx.set_node_attributes(G, dict(enumerate(bid)), "bid")

        seller = 0
        result = stm(G, seller)

        self.assertTrue(result.feasible)
        self.assertEqual(result.revenue, 11)
        self.assertEqual(result.socialWelfare, 13)
        self.assertAlmostEqual(result.efficiencyRatio, 0.7647058823529411)

if __name__ == "__main__":
    unittest.main()