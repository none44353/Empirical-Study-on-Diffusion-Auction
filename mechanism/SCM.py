from random import randint
import networkx as nx
import unittest

try:
    import mechanismBase
    import STM
except ImportError:
    import mechanism.mechanismBase as mechanismBase
    import mechanism.STM as STM
    
class SCM(mechanismBase.DiffusionAuction):
    name = "SCM"
    
    def __init__(self):
        super().__init__()
        self.stm = STM.STM()

    @staticmethod
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

    @staticmethod
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

    def __call__(self, G, seller):
        reachable = nx.descendants(G, seller) | set([seller])
        Gamma = STM.STM.getTopoGamma(G, seller)
        clusterIndex = SCM.getSybilClusterIndex(G, seller, Gamma)
        clustersGraph = nx.DiGraph()
        clustersGraph.add_node(seller)
        for i, j in G.edges():
            if i in reachable and clusterIndex[i] != clusterIndex[j]:
                clustersGraph.add_edge(clusterIndex[i], clusterIndex[j])
        
        clustersTree = SCM.getRSPTree(clustersGraph, seller)
        subG = G.copy()
        for i, j in G.edges():
            if i in reachable and clusterIndex[i] != clusterIndex[j]:
                if not clustersTree.has_edge(clusterIndex[i], clusterIndex[j]): 
                    subG.remove_edge(i, j)
        return self.stm(subG, seller, Gamma)

class TestSCM(unittest.TestCase):
    def testSCM_hand(self):
        scm = SCM()
        G = nx.DiGraph()
        E = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (3, 6), (4, 7), (5, 7), (6, 7), (3, 8), (100, 101)]
        G.add_edges_from(E)
        bid = [0, 2, 3, 5, 7, 13, 11, 17, 19, 23, 29]
        nx.set_node_attributes(G, dict(enumerate(bid)), "bid")

        seller = 0
        result = scm(G, seller)

        self.assertTrue(result.feasible)
        self.assertEqual(result.revenue, 17)
        self.assertEqual(result.socialWelfare, 19)
        self.assertAlmostEqual(result.efficiencyRatio, 1.0)

if __name__ == "__main__":
    unittest.main()