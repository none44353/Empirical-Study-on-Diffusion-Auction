import networkx as nx
from abc import ABC, abstractmethod

class GraphGen(ABC):
    @abstractmethod
    def __call__(self, n: int) -> nx.DiGraph:
        pass