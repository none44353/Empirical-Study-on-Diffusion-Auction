import networkx as nx
import scipy.stats as stats
import random
from typing import Tuple, Any
import numpy as np
from zipfian import ZipfGenerator

from mechanism.mechanismBase import DiffusionAuction
from mechanism.IDM import IDM
from mechanism.STM import STM
from mechanism.NSP import NSP
from mechanism.SCM import SCM

from graphgen.genBase import GraphGen
from graphgen.gnp import GNP
from graphgen.prices import Price_s
# from graphgen.Schweimer22 import Schweimer22
from graphgen.staticFile import StaticFile
from multiprocessing import Pool

# mode = 'FAST'
mode = 'STANDARD'

TEST_TIMES = 10

test_mechanisms = {
    'NSP': NSP(),
    'IDM': IDM(),
    'STM': STM(),
    'SCM': SCM()
}

test_graphs = {
    'GNP p=0.1 n=30': (GNP(p = 0.1), 30),
    'GNP p=0.05 n=100': (GNP(p = 0.05), 100),
    'GNP p=0.003 n=2000': (GNP(p = 0.01), 2000),
    #"Price's m=3 c=1 gamma=1 n=30": (Price_s(m = 3, c = 1, gamma = 1), 30),
    #"Price's m=5 c=1 gamma=1 n=30": (Price_s(m = 5, c = 1, gamma = 1), 30),
    #"Price's m=6 c=1 gamma=1 n=100": (Price_s(m = 6, c = 1, gamma = 1), 100),
    #"Price's m=15 c=1 gamma=1 n=100": (Price_s(m = 15, c = 1, gamma = 1), 100),
    #"Price's m=12 c=1 gamma=1 n=500": (Price_s(m = 12, c = 1, gamma = 1), 500),
    #"Price's m=20 c=1 gamma=1 n=500": (Price_s(m = 20, c = 1, gamma = 1), 500),
    #'Schweimer22 GlobaldevII': (StaticFile('data/static_graph/GlobaldevII.gpickle'), 459),
    #'Schweimer22 datamining': (StaticFile('data/static_graph/datamining.gpickle'), 2013),
    #'Schweimer22 Calfire': (StaticFile('data/static_graph/Calfire.gpickle'), 3580),
    #'Schweimer22 Bioinformatics': (StaticFile('data/static_graph/Bioinformatics.gpickle'), 6003),
    #'Schweimer22 Vegan': (StaticFile('data/static_graph/Vegan.gpickle'), 11015),
}

test_distributions = {
    'uniform': stats.uniform(loc=0, scale=1),
    'powerlaw': ZipfGenerator(alpha=1, n=1000),
    'powerlaw2': stats.zipf(a=2),
    'halfnorm': stats.halfnorm(),
}

if mode == 'FAST':
    test_graphs = {
        x: y for x, y in test_graphs.items() if y[1] < 500
    }

pregenerated_graphs = {name: [] for name in test_graphs}
pregenerated_seller = {name: [] for name in test_graphs}
pregenerated_bids = {name: [] for name in test_distributions}

def test(mechanism: DiffusionAuction, 
        gname: str, 
        dname: str,
        i: int) -> DiffusionAuction.MechanismResult:
    graph = pregenerated_graphs[gname][i].copy()
    nodes = list(graph.nodes)
    bids = dict(zip(nodes, pregenerated_bids[dname][i]))
    seller = pregenerated_seller[gname][i]
    bids[seller] = 1e-8
    nx.set_node_attributes(graph, bids, 'bid')
    return mechanism(graph, seller)

def init():
    max_n = 0
    for gname in test_graphs:
        for i in range(TEST_TIMES):
            print(f'Pregenerating graph {gname} {i}')
            graphconf = test_graphs[gname]
            graphgen, n = graphconf
            graph = graphgen(n)
            max_n = max(max_n, len(graph.nodes))
            pregenerated_graphs[gname].append(graph)
            degree_threshold = graph.number_of_edges() / len(graph.nodes)
            deg = dict(graph.degree())
            out_deg = dict(graph.out_degree())
            pregenerated_seller[gname].append(random.choice([x for x in graph.nodes if deg[x] >= degree_threshold and out_deg[x] >= 1]))
    
    for dname in test_distributions:
        print(f'Pregenerating bid {dname}')
        distribution = test_distributions[dname]
        for i in range(TEST_TIMES):
            pregenerated_bids[dname].append(distribution.rvs(max_n))

def main():
    for gname in test_graphs:
        for dname in test_distributions:
            for mname, mechanism in test_mechanisms.items():
                results = list(map(lambda i: test(mechanism, gname, dname, i), list(range(TEST_TIMES))))
                eff_ratio = list(map(lambda x: x.efficiencyRatio, results))
                normalized_revenue = list(map(lambda x: x.normalizedRevenue, results))
                print(f'{gname} {mname} {dname}, efficiency ratio {np.mean(eff_ratio):.4f}, normalized revenue {np.mean(normalized_revenue):.4f}')

if __name__ == '__main__':
    init()
    main()