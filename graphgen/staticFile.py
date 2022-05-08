import networkx as nx
from .genBase import GraphGen

class StaticFile(GraphGen):
    def __init__(self, filename: str):
        self.filename = filename
        self.graph = nx.read_gpickle(self.filename)

    def __call__(self, _) -> nx.DiGraph:
        return self.graph.copy()