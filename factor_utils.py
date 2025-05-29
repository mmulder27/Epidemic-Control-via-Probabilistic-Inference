import numpy as np
from copy import deepcopy
from pgmpy.models import FactorGraph
from pgmpy.factors.discrete import DiscreteFactor
from collections import defaultdict

class FastDiscreteFactor(DiscreteFactor):
    # define faster hash function, depending only on variable names
    def __hash__(self):
        variable_hashes = [hash(variable) for variable in self.variables]
        return hash(sum(variable_hashes))
        # return hash(str(sorted(variable_hashes)))

class EpidemicFactorGraph(FactorGraph):
    def __init__(self, Params=None, contacts=None, tests=None):
        super().__init__()
        self.Params = Params
        self.contacts = contacts
        self.tests = tests
        self.card_dict = {}
        self._build_graph()

    def _build_graph(self):
        # 1) Set up data structures for creating variables and factors
        contact_times = defaultdict(list)  # (i,j) -> list of times that individuals i and j had contact
        X_i = defaultdict(set)             # i -> set of all times that individual i had contact with another individual
        
        for i, j, t, λ in self.contacts:
            contact_times[(i, j)].append(t)
            X_i[i].add(t)
            X_i[j].add(t)

        # 2) Remove duplicates and sort
        to_delete = set()
        for (i, j) in list(contact_times):
            contact_times[(i, j)] = sorted(set(contact_times[(i, j)]))
            if (j, i) in contact_times and i > j:
                to_delete.add((i, j))

        for key in to_delete:
            del contact_times[key]
        
        for node in X_i:
            X_i[node] = sorted(X_i[node])
        
        # 3) Create s_ij variables for each contact event
        self.edge_index = defaultdict(int)
        for (i, j), times in contact_times.items():
            for idx, t in enumerate(times):
                var_name = f"(s_{i}_{j}_, s_{j}_{i})_{idx}"
                self.add_node(var_name)
                # cardinality = (#contact events for i and j) + 1
                self.card_dict[var_name] = len(times) + 1
            # record how many events we added, so we can reference them later
            self.edge_index[(i, j)] = len(times)
        
        # 4) Create t_i and r_i for each individual
        nodes = set(i for i,_,_,_ in self.contacts) | set(j for _,j,_,_ in self.contacts)
        for i in nodes:
            t_var = f"t_{i}"
            r_var = f"r_{i}"
            self.add_node(t_var)
            self.add_node(r_var)
            # cardinality = |X_i| + 1
            self.card_dict[t_var] = len(X_i[i]) + 1
            self.card_dict[r_var] = len(X_i[i]) + 1
    
    # 5) (Later) you’ll hook up all the factors and edges:
    #    - Transmission factors on (s_{i}_{j}_{k}, t_i, r_i)
    #    - Recovery factors on (r_i, t_i)
    #    - Auto‐infection factors on a virtual s_{i*}_{i} per i
    #    - Observation factors on (t_i, r_i)
    #
    #    And for each factor phi, you’ll do:
    #        for v in phi.scope():
    #            self.add_edge(v, phi)



        # 4) now add all the factors
        self._add_transmission_factors()
        self._add_recovery_factors()
        self._add_auto_infection_factors()
        self._add_observation_factors()




class SimpleFactorGraph(FactorGraph):
    # simplify some methods and error-checking in FactorGraph for efficiency
    def __init__(self):
        super(SimpleFactorGraph, self).__init__()
        self.card_dict = {}

    def get_variable_nodes(self):
        variable_nodes = set([x for factor in self.factors for x in factor.scope()])
        return list(variable_nodes)

    def add_factors(self, *factors):
        for factor in factors:
            self.factors.append(factor)
            for var in factor.scope():
                self.card_dict[var] = factor.get_cardinality([var])[var]

    def remove_node(self, f):
        super().remove_node(f)
        if f in self.card_dict.keys():
            del self.card_dict[f]

    def get_cardinality(self, variable_name=None):
        """
        Returns the cardinality of the node or a dict over all nodes if not specified.
        """
        if variable_name is not None:
            return self.card_dict[variable_name]
        else:
            return self.card_dict

    def add_evidence(self, var, val):
        """
        ADD_EVIDENCE - Adds the "evidence" that variable "var" takes
        on value "val".  This slices the factors neighboring "var" accordingly
        and returns the updated factor graph structure.

        Parameters
        ----------
        G : Factor Graph
        var : variable node of the factor graph
        val : value of variable evidence

        Returns
        -------
        G : modified factor graph involving evidence information
        """
        var_dim = self.get_cardinality(var)
        # iterate factor neighbors
        fac_nbrs = deepcopy(list(self.neighbors(var)))

        for fac_nbr in fac_nbrs:
            potential = fac_nbr.values
            var_nbrs = fac_nbr.scope()
            card = fac_nbr.cardinality

            self.remove_factors(fac_nbr)

            if len(var_nbrs) == 1:
                continue
                # new_var_nbrs = var_nbrs
                # new_card = card
                # new_potential = np.zeros(card[0])
                # new_potential[val] = 1
            else:
                I = var_nbrs.index(var)
                ind = [slice(None)] * len(var_nbrs)
                ind[I] = val
                new_potential = potential[tuple(ind)]
                var_nbrs.remove(var)
                new_var_nbrs = var_nbrs
                l_card = list(card)
                l_card.pop(I)
                new_card = np.array(l_card)
            phi = FastDiscreteFactor(new_var_nbrs, new_card, new_potential)
            if phi in self.factors:
                continue

            self.add_factors(phi)
            self.add_nodes_from([phi])
            edges = []
            for variable in new_var_nbrs:
                edges.append((variable, phi))
            self.add_edges_from(edges)

        # marginal = np.zeros(var_dim)
        # marginal[val] = 1
        # phi = FastDiscreteFactor([var], [var_dim], marginal)
        # self.add_factors(phi)
        # self.add_nodes_from([phi])
        # self.add_edges_from([(var, phi)])
        self.remove_node(var)

        return self

    def add_evidences(self, vars, vals):
        for var, val in zip(vars, vals):
            self.add_evidence(var, val)




def belief_diff(b1, b2):
    """
    BELIEF_DIFF - Computes the symmetric L1 distance between beliefs 'b1'
    and 'b2'. 
    
    Parameters
    ----------
    b1, b2 : N-entry dictionaries of variables and their corresponding marginals.

    Returns
    -------
    D : N-dimensional numpy array of L1 distances.
    """

    num_b = len(b1)
    D = np.zeros(num_b)
    for i, key in enumerate(b1.keys()):
        D[i] = 0.5 * np.sum(abs(b1[key] - b2[key]))

    return D
