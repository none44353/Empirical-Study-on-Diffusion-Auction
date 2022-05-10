from IDM import IDM
from NSP import NSP
from STM import STM
from SCM import SCM
from mechanismBase import DiffusionAuction
import unittest
import networkx as nx

class TestTrivialGraph(unittest.TestCase):
    def test_trivial(self):
        G = nx.DiGraph()
        E = [(0, 1)]
        G.add_edges_from(E)

        bid = [1, 1e-8]
        nx.set_node_attributes(G, dict(enumerate(bid)), "bid")
        seller = 1

        for mechanism in [NSP(), IDM(), STM(), SCM()]:
            print(mechanism.name)
            result = mechanism(G, seller)
            self.assertTrue(result.feasible)
            self.assertAlmostEqual(result.socialWelfare, 0)
            self.assertEqual(result.revenue, 0)
            self.assertAlmostEqual(result.efficiencyRatio, 0)
        
        bid = [1e-8, 1]
        nx.set_node_attributes(G, dict(enumerate(bid)), "bid")
        seller = 0

        for mechanism in [NSP(), IDM(), STM(), SCM()]:
            print(mechanism.name)
            result = mechanism(G, seller)
            self.assertTrue(result.feasible)
            self.assertEqual(result.winner, 1)
            self.assertAlmostEqual(result.socialWelfare, 1)
            self.assertAlmostEqual(result.revenue, 0)
            self.assertAlmostEqual(result.efficiencyRatio, 1)
        

if __name__ == "__main__":
    unittest.main()