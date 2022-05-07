import networkx as nx
from genBase import GraphGen

class Price_s(GraphGen):
    def __init__(self, p: float):
        self.p = p

    def __call__(self, n: int) -> nx.DiGraph:
        raise NotImplementedError()