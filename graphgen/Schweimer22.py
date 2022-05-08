# The majority of this code is based on the following work:
#   https://gitlab.com/eu_hidalgo/use_cases/-/tree/master/social_network/network_generation
# The algorithm is present in the following paper:
#   Christoph Schweimer, Christine Gfrerer, Florian Lugstein, David Pape, 
#     Jan A. Velimsky, Robert Elsässer, and Bernhard C. Geiger. 2022. 
#   Generating Simple Directed Social Network Graphs for Information Spreading. 
#   In Proceedings of the ACM Web Conference 2022 (WWW '22). 
#   Association for Computing Machinery, New York, NY, USA, 1475–1485. 
#   https://doi.org/10.1145/3485447.3512194

import numpy as np
import pandas as pd
import networkx as nx
import random
from scipy.stats import chi2
from scipy.stats import norm
from itertools import combinations

from .genBase import GraphGen

class Schweimer22(GraphGen):
    def __init__(self,
        Rho_1: float = .8, Rho_2: float = .8, Rho_3: float = .8,
        rec_df: float = 0.5, rec_loc: float = 0.49, rec_scale: float = 20,
        in_df: float = 0.5, in_loc: float = 0.49, in_scale: float = 10,
        out_df: float = 0.2, out_loc: float = 0.49, out_scale: float = 25,
        REWIRE: float = 1
        ):
        self.Rho_1 = Rho_1
        self.Rho_2 = Rho_2
        self.Rho_3 = Rho_3
        self.rec_df = rec_df
        self.rec_loc = rec_loc
        self.rec_scale = rec_scale
        self.in_df = in_df
        self.in_loc = in_loc
        self.in_scale = in_scale
        self.out_df = out_df
        self.out_loc = out_loc
        self.out_scale = out_scale
        self.REWIRE = REWIRE

    
    def __call__(self, n: int) -> nx.DiGraph:
        R_1 = 2*np.sin(self.Rho_1*np.pi/6)
        R_2 = 2*np.sin(self.Rho_2*np.pi/6)
        R_3 = 2*np.sin(self.Rho_3*np.pi/6)
        mean = [0,0,0]
        cov = [[1,R_1,R_2], [R_1,1,R_3], [R_2,R_3,1]] 

        norm_1,norm_2,norm_3 = np.random.multivariate_normal(mean, cov, n).T

        # Transform the data to be uniformly distributed
        unif_1 = norm.cdf(norm_1)
        unif_2 = norm.cdf(norm_2)
        unif_3 = norm.cdf(norm_3)

        RECIPROCAL = chi2.ppf(unif_1, df=self.rec_df, loc=self.rec_loc, scale=self.rec_scale)
        RECIPROCAL = np.round(RECIPROCAL)
        ONLY_IN = chi2.ppf(unif_2, df=self.in_df, loc=self.in_loc, scale=self.in_scale)
        ONLY_IN = np.round(ONLY_IN)
        ONLY_OUT = chi2.ppf(unif_3, df=self.out_df, loc=self.out_loc, scale=self.out_scale)
        ONLY_OUT = np.round(ONLY_OUT)

        SAMPLED = pd.DataFrame(columns=['Reciprocal', 'In', 'Out'])

        SAMPLED['Reciprocal'] = RECIPROCAL
        SAMPLED['In'] = ONLY_IN
        SAMPLED['Out'] = ONLY_OUT

        # Get the total degree per node
        total = SAMPLED['Reciprocal'] + SAMPLED['In'] + SAMPLED['Out']
        SAMPLED['Total'] = total

        # Shuffle the dataframe
        SAMPLED = SAMPLED.sample(frac=1)

        # Number of nodes
        numNodes = SAMPLED.shape[0]

        # Node numbers
        Nodes = list(range(1, numNodes+1))

        # Reciprocal degrees
        RECIPROCAL = list(SAMPLED['Reciprocal'])

        # Out degree without reciprocal edges
        OUT = list(SAMPLED['Out'])

        # In degree without reciprocal edges
        IN = list(SAMPLED['In'])

        # Number of nodes with a reciprocal degree bigger than 0 
        reciprocal = [i for i in RECIPROCAL if i > 0]
        reciprocal = len(reciprocal)

        # Number of nodes with an out degree bigger than 0 
        out = [i for i in OUT if i > 0]
        out = len(out)

        # Number of nodes with an in degree bigger than 0 
        r = [i for i in IN if i > 0]
        r = len(r)

        # Number of reciprocal edges
        Edges_m = sum(RECIPROCAL)*2

        # Number of directed edges
        Edges_d = (sum(OUT) + sum(IN))/2

        diag_sum_m = 0

        # Probability of the diagonal elements
        for i in range(numNodes):
            if RECIPROCAL[i] != 0:
                diag_sum_m += RECIPROCAL[i]**2/Edges_m
                
        # Distribute the sum uniformly over all other possible edges
        diag_sum_m_dist = diag_sum_m/(reciprocal*(reciprocal-1)/2)

        diag_sum_d = 0
        diag = 0

        # Probability of the diagonal elements
        for i in range(numNodes):
            if (IN[i] != 0 and OUT[i] != 0):
                diag_sum_d += IN[i]*OUT[i]/Edges_d
                diag +=1
                
        # Distribute the sum uniformly over all other possible edges
        diag_sum_d_dist = diag_sum_d/(out*r-diag)

        Connection_r = []

        counter = 0
        directed_sum = 0

        for i in range(numNodes):      
            for j in range(i+1, numNodes):
                a = 0
                
                # Sampling of reciprocal edges
                if (RECIPROCAL[j] != 0 and RECIPROCAL[i] != 0):
                    pr = 2*RECIPROCAL[i]*RECIPROCAL[j]/Edges_m + diag_sum_m_dist
                    if pr>1:
                        pr=1
                    a = np.random.choice(np.arange(0,2), p=[1-pr, pr])
                    if a == 1:
                        Connection_r.append((i+1,j+1))
                        Connection_r.append((j+1,i+1))
                        
                        # This part is for the compensation for the sampled edges that we lose while trying to sample directed edges
                        if(IN[i] != 0 and OUT[j] != 0):
                            counter += 1
                            directed_sum += IN[i]*OUT[j]/Edges_d + diag_sum_d_dist
                            
                        if(IN[j] != 0 and OUT[i] != 0):
                            counter += 1
                            directed_sum += OUT[i]*IN[j]/Edges_d + diag_sum_d_dist
        
        sampled_reciprocal = directed_sum/(out*r-diag-counter)

        Connection_d = []

        for i in range(numNodes):      
            for j in range(i+1, numNodes):
                # if a reciprocal edge is sampled a = 1, we do not get down here
                    
                # Sampling of directed edges (bigger number to smaller number)
                if(IN[i] != 0 and OUT[j] != 0):
                    pr_d = IN[i]*OUT[j]/Edges_d + diag_sum_d_dist + sampled_reciprocal
                    if pr_d>1:
                        pr_d=1
                    b = np.random.choice(np.arange(0,2), p=[1-pr_d, pr_d])
                    if b == 1:
                        Connection_d.append((j+1,i+1))

                # Sampling of directed edges (smaller number to bigger number)
                if(IN[j] != 0 and OUT[i] != 0):
                    pr_d = OUT[i]*IN[j]/Edges_d + diag_sum_d_dist + sampled_reciprocal
                    if pr_d>1:
                        pr_d=1
                    c = np.random.choice(np.arange(0,2), p=[1-pr_d, pr_d])
                    if c == 1:
                        Connection_d.append((i+1,j+1))

        # Directed empty graph
        Graph = nx.DiGraph()

        # Add the nodes
        Graph.add_nodes_from(Nodes)

        # Add the edges
        Graph.add_edges_from(Connection_r)   
        Graph.add_edges_from(Connection_d)   

        # Translate the graph into a matrix
        M = nx.to_numpy_matrix(Graph)

        # Multiply with the transpose to find reciprocal edges
        Reciprocal = nx.from_numpy_matrix(np.multiply(M,(M.T)), create_using=nx.DiGraph())

        # Re-numerate the nodes of the reciprocal graph
        mapping = {i: Graph for i, Graph in enumerate(list(Graph.nodes()))}
        Reciprocal = nx.relabel_nodes(Reciprocal, mapping)

        # List of reciprocal edges
        reciprocal_edges = list(Reciprocal.edges)

        # Save the node information for the graph in a dataframe
        degrees = pd.concat([pd.Series(dict(Graph.in_degree()), name='in'), 
                            pd.Series(dict(Graph.out_degree()), name='out'),
                            pd.Series(dict(Reciprocal.in_degree()), name='reciprocal'),
                            pd.Series(pd.Series(dict(Graph.in_degree()))+pd.Series(dict(Graph.out_degree())), name='total')], 
                            axis=1)

        degrees['Ne'] = degrees['total'] - degrees['reciprocal']

        # Nodes with 0 or 1 neighbors are not considered for the rewiring procedure
        SORT = sorted(list(degrees['Ne']))

        if self.REWIRE == 0:
            DEG = 0
        else:
            if (int(np.round(numNodes*self.REWIRE)) == numNodes):
                DEG = SORT[int(np.round(numNodes*self.REWIRE))-1]
            else:
                DEG = SORT[int(np.round(numNodes*self.REWIRE))-1]

        # Nodes with a small reciprocal degree
        nodes = list(degrees[(degrees['Ne'] > 1) & (degrees['Ne'] <= DEG)].index)

        MED = degrees[(degrees['Ne'] > 1) & (degrees['Ne'] <= DEG)]['Ne'].median()

        for i in range(len(nodes)):
            help = 0

            # First degree neighbors of the node we are investigating
            # As it is a directed graph, we need the neighbors and concatenate the predecessors
            Ne1 = list(set(list(Graph.predecessors(nodes[i])) + list(Graph.neighbors(nodes[i]))))

            # Second degree neighbors of the nodes we are investigating
            Ne2 = []
            dir_degree = len(Ne1)
            if dir_degree <= MED:
                x = 0.5
            else:
                x = 0.3
                
            # Get all the first degree neighbor pairs
            output = sum([list(map(list, combinations(Ne1, 2)))], [])
            Neighbors = random.sample(output, int(np.ceil(dir_degree * (dir_degree-1) * x)))
            
            # We give a fraction of the first degree neighbor pairs the chance to connect, depending on the node degree
            for j in range(int(np.ceil(dir_degree * (dir_degree-1) * x))):
                
                help = 0
            
                # Randomly sample the first degree neighbors that we want to check for rewiring
                # Note that the same pair can be sampled again
                Neighbor1 = Neighbors[j][0]
                Neighbor2 = Neighbors[j][1]

                # Candidates for rewiring (2nd degree neighbors)
                Candidates = []

                # Rewiring only possible if the list of candidates is not empty
                Candidates_final = []

                # Final edge to rewire
                rewire = 0

                # Only potentially rewire if the two nodes are not connected (these already form a triangle)
                if ((Neighbor1, Neighbor2) in Graph.edges or (Neighbor2, Neighbor1) in Graph.edges) == False:
                    # Second degree neighbors: Here we could have multipe entries for one node as we can have a reciprocal edge
                    # Therefore we eliminate duplicate entries by making the list a set and then a list again
                    Ne2 = []
                    Ne2.append(list(set(list(Graph.predecessors(Neighbor1)) + list(Graph.neighbors(Neighbor1)))))
                    Ne2.append(list(set(list(Graph.predecessors(Neighbor2)) + list(Graph.neighbors(Neighbor2)))))
                        
                    # Remove all 2nd degree neighbors with a smaller number than the currently investigated node
                    # This is to not interrupt the already created triangles too much
                    Ne2[0] = [x for x in Ne2[0] if x > nodes[i]]
                    Ne2[1] = [x for x in Ne2[1] if x > nodes[i]]

                    # Find the nodes in Ne2 that are connected to both nodes of Ne1 and remove them
                    non_unique = []
                    intersect = set(Ne2[0]).intersection(set(Ne2[1]))
                    for m in range(len(intersect)):
                        non_unique.append(list(intersect)[m])

                    # Remove the non-unique nodes from the list of neighbors (includes the node we are originally investigating)
                    for m in range(len(non_unique)):
                        Ne2[0].remove(non_unique[m])
                        Ne2[1].remove(non_unique[m])

                    # Lists may not be empty
                    if (Ne2[0] and Ne2[1]):

                        # Give it 10 tries to find an edge for the rewiring (here we do not loop throigh all, but randomy choose two nodes from both lists of 2nd degree neighbors)
                        for l in range(10):
                            
                            # With the variable 'help', we check if we have found a match for rewiring
                            if help == 0:
                                
                                rewire = 0

                                # Randomly choose a node from Ne2[0] and from Ne2[1]
                                Candidate1 = random.choice(Ne2[0])
                                Candidate2 = random.choice(Ne2[1])

                                # Save the candidate edges in a list
                                Candidates = [(Candidate1, Candidate2), (Candidate2, Candidate1)]

                                # There may not be a connection between them
                                if (Candidates[0] not in Graph.edges and Candidates[1] not in Graph.edges):
                                    rewire = Candidates[0]

                                    # In the variable "rewire", we save the edges that can be established
                                    if rewire != 0:

                                        # rewire[0] - Neighbor1 and Neighbor2 - rewire[1]
                                        if ((rewire[0], Neighbor1) in Graph.edges and (Neighbor1, rewire[0]) in Graph.edges and (rewire[1], Neighbor2) in Graph.edges and (Neighbor2, rewire[1]) in Graph.edges):
                                            Graph.remove_edge(rewire[0], Neighbor1)
                                            Graph.remove_edge(Neighbor1, rewire[0])
                                            Graph.remove_edge(rewire[1], Neighbor2)
                                            Graph.remove_edge(Neighbor2, rewire[1])

                                            Graph.add_edge(rewire[0], rewire[1])
                                            Graph.add_edge(rewire[1], rewire[0])
                                            Graph.add_edge(Neighbor1, Neighbor2)
                                            Graph.add_edge(Neighbor2, Neighbor1)
                                            help = 1

                                        # rewire[0] -> Neighbor1 and Neighbor2 -> rewire[1]
                                        if ((rewire[0], Neighbor1) in Graph.edges and (Neighbor1, rewire[0]) not in Graph.edges and (rewire[1], Neighbor2) not in Graph.edges and (Neighbor2, rewire[1]) in Graph.edges):
                                            Graph.remove_edge(rewire[0], Neighbor1)
                                            Graph.remove_edge(Neighbor2, rewire[1])

                                            Graph.add_edge(rewire[0], rewire[1])
                                            Graph.add_edge(Neighbor2, Neighbor1)
                                            help = 1

                                        # rewire[0] <- Neighbor1 and Neighbor2 <- rewire[1]    
                                        if ((rewire[0], Neighbor1) not in Graph.edges and (Neighbor1, rewire[0]) in Graph.edges and (rewire[1], Neighbor2) in Graph.edges and (Neighbor2, rewire[1]) not in Graph.edges):
                                            Graph.remove_edge(Neighbor1, rewire[0])
                                            Graph.remove_edge(rewire[1], Neighbor2)

                                            Graph.add_edge(rewire[1], rewire[0])
                                            Graph.add_edge(Neighbor1, Neighbor2)
                                            help = 1
        
        return Graph
