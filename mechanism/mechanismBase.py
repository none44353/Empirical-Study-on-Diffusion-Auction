import networkx as nx
import numpy as np
from abc import ABC, abstractmethod
from typing import Any, List, Dict, Tuple
import abstractcp as acp
import unittest

class DiffusionAuction(ABC, acp.Abstract):
    class MechanismResult:
        def __init__(self, seller: Any, winner: Any, monetaryTransfer: Dict[Any, float], G: nx.DiGraph):
            self.seller = seller
            self.winner = winner
            self.monetaryTransfer = monetaryTransfer
            self.G = G
        
        @property
        def feasible(self) -> bool:
            return self.winner in self.G.nodes \
                and sum(self.monetaryTransfer.values()) == 0 \
                and all(self.monetaryTransfer[x] == 0 for x in self.G.nodes - nx.descendants(self.G, self.seller) - set([self.seller]))

        @property
        def revenue(self) -> float:
            return self.monetaryTransfer[self.seller]

        @property
        def socialWelfare(self) -> float:
            return self.G.nodes[self.winner]["bid"]
        
        @property
        def efficiencyRatio(self) -> float:
            return self.socialWelfare / getOptimal(self.G, self.seller)
        
        @property
        def normalizedRevenue(self) -> float:
            return self.revenue / getOptimal(self.G, self.seller)
    
    name: str = acp.abstract_class_property(str)

    @abstractmethod
    def __call__(self, G: nx.DiGraph, seller: Any) -> MechanismResult:
        pass

def getOptimal(G: nx.DiGraph, seller) -> float:
    reachableNodes = nx.descendants(G, seller) | set([seller])
    return max([G.nodes[i]["bid"] for i in reachableNodes])

def getAverageBid(G: nx.DiGraph, seller) -> float:
    reachableNodes = nx.descendants(G, seller) | set([seller])
    return np.mean([G.nodes[i]["bid"] for i in reachableNodes])

class TestOptimal(unittest.TestCase):
    def test_getOptimal(self):
        G = nx.DiGraph()
        E = [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5), (2, 5), (3, 6), (5, 7), (8, 9)]
        G.add_edges_from(E)
        bids = [9, 9, 8, 2, 4, 4, 3, 5, 3, 2]
        nx.set_node_attributes(G, dict(enumerate(bids)), "bid")
        self.assertEqual(getOptimal(G, 0), 9)
        self.assertEqual(getOptimal(G, 5), 5)
        self.assertEqual(getOptimal(G, 8), 3)
        self.assertEqual(getOptimal(G, 9), 2)

if __name__ == "__main__":
    unittest.main()