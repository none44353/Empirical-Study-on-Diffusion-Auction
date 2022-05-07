import networkx as nx
from mechanismBase import DiffusionAuction
import unittest

class IDM(DiffusionAuction):
    @staticmethod
    def getPrice(G, seller, bid, idom, H):
        # H is the set of reachable vertexes from seller
        dist = nx.single_source_shortest_path_length(G, seller)
        domTree = nx.DiGraph()
        for i in H:
            if i != seller: 
                domTree.add_edge(idom[i], i)

        H.sort(key = (lambda i: dist[i]))
        #the maximum bid of i's subtree in domTree Graph, denoted as maxbid
        maxbid = {}
        for i in reversed(H):
            maxbid[i] = max([maxbid[j] for j in domTree.successors(i)] + [bid[i]])
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
                price[adj[jx]] = max(price[i], bid[i], preMaxVal[jx], sufMaxVal[jx])

        return price, maxbid

    @staticmethod
    def getDiffSeq(seller, reachableNodes, idom, bid):
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

    def __call__(self, G, seller):
        bid = G.nodes.data("bid")
        idom = nx.immediate_dominators(G, seller)
        reachableNodes = list(nx.descendants(G, seller)) + [seller]
        price, maxAlpha = IDM.getPrice(G, seller, bid, idom, reachableNodes)
        mxBidder, diffusionSeq = IDM.getDiffSeq(seller, reachableNodes, idom, bid)

        winner = -1
        monetaryTransfer = dict(zip(G.nodes, [0] * len(G.nodes)))
        for ix, i in enumerate(diffusionSeq):
            if (i == mxBidder) or (price[diffusionSeq[ix + 1]] == bid[i]):
                winner = i
                monetaryTransfer[i] = -price[i]
                break
            else:
                monetaryTransfer[i] = price[diffusionSeq[ix + 1]] - price[i]

        return DiffusionAuction.MechanismResult(seller, winner, monetaryTransfer, G)

class TestIDM(unittest.TestCase):
    def testIDM_hand(self):
        mechanism = IDM()
        G = nx.DiGraph()
        E = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 5), (3, 6), (5, 7), (8, 9)]
        G.add_edges_from(E)
        # load bids of agents
        bid = [0, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        seller = 0
        nx.set_node_attributes(G, dict(enumerate(bid)), "bid")

        result = mechanism(G, seller)

        self.assertTrue(result.feasible())
        self.assertEqual(result.revenue(), 13)
        self.assertEqual(result.socialWelfare(), 17)
        self.assertAlmostEqual(result.efficiencyRatio(), 1.0)
    
    def testIDM_Li2017Fig2(self):
        mechanism = IDM()
        G = nx.DiGraph()
        E = [('s', 'A'), ('s', 'B'), ('s', 'C'), ('C', 'J'), ('C', 'H'),
             ('A', 'D'), ('B', 'E'), ('C', 'E'), ('C', 'I'), ('H', 'I'),
             ('D', 'E'), ('I', 'L'), ('D', 'F'), ('D', 'G'), ('G', 'K')]
        G.add_edges_from(E)
        G.add_edges_from([(y, x) for x, y in E])
        bid = {'s': 0, 'A': 1, 'B': 3, 'C': 2, 'J': 4, 'H': 11, 
            'I': 12, 'D': 5, 'E': 7, 'F': 10, 'G': 8, 'K': 6, 'L': 13}
        nx.set_node_attributes(G, bid, "bid")
        seller = 's'
        result = mechanism(G, seller)
        self.assertTrue(result.feasible())
        self.assertEqual(result.winner, 'I')
        self.assertEqual(result.monetaryTransfer['I'], -11)
        self.assertEqual(result.monetaryTransfer['C'], 1)

if __name__ == "__main__":
    unittest.main()