import networkx as nx
from genBase import GraphGen

class GNP(GraphGen):
    def __init__(self, p: float):
        self.p = p

    def __call__(self, n: int) -> nx.DiGraph:
        return nx.fast_gnp_random_graph(n, self.p, seed=None, directed=True)