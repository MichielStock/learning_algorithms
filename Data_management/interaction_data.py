"""
Created on Fri Apr 22 2016
Last update: Sat Apr 23 2016

@author: Michiel Stock
michielfmstock@gmail.com

Module for storing interaction datasets

Each interaction dataset has following properties:
    - metadata: name, reference, category, interaction types
    - columns: kind, number, datatype, (list of objects)
    - rows: kind, number, datatype, (list of objects)
    - interaction (adjacency) matrix
    - version number
"""

import json
import numpy as np

def dense_graph_to_matrix(shape, graph, dtype):
    """
    Turns dense graph into a numpy matrix
    """
    Y = np.zeros(shape, dtype=dtype)
    for i, vals in graph.items():
        # keys in json are str
        Y[int(i), :] = np.array(vals)[:]
    return Y

def sparse_graph_to_matrix(shape, graph, dtype):
    Y = np.zeros(shape, dtype=dtype)
    for i, nodes in graph.items():
        for j, val in nodes:
            # keys in json are str
            Y[int(), j] = val
    return Y

class InteractionDataset:
    """
    Class to store interaction datasets

    Each interaction dataset has following properties:
        - metadata: name, reference, category, interaction types
        - columns: kind, number, datatype, (list of objects)
        - rows: kind, number, datatype, (list of objects)
        - interaction (adjacency) matrix
        - version number

    Here we use the convention that 'u' refers to objects of the rows and 'v'
    to objects of the columns.
    """
    def __init__(self, interaction_matrix, name=None, reference=None,
                    category=None, interaction_type=None, u_objects=None,
                    u_type=None, v_objects=None, v_type=None):
        # interaction matrix
        self.Y = interaction_matrix  # e.g. a protein-ligand interaction binding
        # matrix

        # metadata
        self.name = name  # e.g. 'Karaman dataset'
        self.reference = reference  # i.e. ref to original paper
        self.category = category  # e.g. 'protein-ligans'
        self.interaction_type = interaction_type  # e.g. 'protein interaction'
        self.data_type = self.Y.dtype

        # information about the rows
        self.u_objects = u_objects
        self.u_type = u_type

        # information about the columns
        self.v_objects = v_objects
        self.v_type = v_type

        self.n_rows, self.n_cols = self.Y.shape

        self.version = 1

    def make_graph(self, dense=False):
        """
        Transforms the interaction dataset into a graph representation
        each row is a key of a dictionary, with either a list of all
        interactions (dense=True) or a list of tuples (col_ind, val) for the
        non-negative interactions (dense=False, default)
        """
        if dense == True:
            graph = {i : list(self.Y[i])
            for i in range(self.n_rows)}
        else:
            graph = {i : [(j, self.Y[i, j])
            for j in range(self.n_cols)
            if self.Y[i, j] != 0]
            for i in range(self.n_rows)}
        return graph

    def dump(self, filename, dense=False, indent=None):
        """
        Saves the dataset into a json file
        Inputs:
            - filname
            - dense (bool) to store the data in dense or sparse format
            - indent (int) indent to generate a readable file (default None)
        """
        fh = open(filename, 'w')
        data = {'version' : self.version,
                'metadata' : {'name' : self.name,
                            'reference' : self.reference,
                            'category' : self.category,
                            'interaction_type' : self.interaction_type,
                            'data_type' : str(self.data_type),
                            'storage' : 'dense' if dense else 'sparse'},
                'rows' : {'objects' : self.u_objects,
                            'type' : self.u_type,
                            'number' : self.n_rows},
                'columns' : {'objects' : self.v_objects,
                            'type' : self.v_type,
                            'number' : self.n_cols},
                'interactions' : self.make_graph(dense)
                }
        json.dump(data, fh, indent=indent, separators=(',', ':'))

    @classmethod
    def load(self, filename):
        fh = open(filename, 'r')
        data = json.load(fh)
        assert data['version'] == 1
        # load interaction matrix
        shape = (data['rows']['number'], data['columns']['number'])
        dtype = data['metadata']['data_type']
        graph = data['interactions']
        if data['metadata']['storage'] == 'dense':
            Y = dense_graph_to_matrix(shape, graph, dtype)
        elif data['metadata']['storage'] == 'sparse':
            Y = sparse_graph_to_matrix(shape, graph, dtype)
        else:
            raise KeyError('storage format is either \'dense\' or \'sparse\'')
        return self(Y)

if __name__ == '__main__':

    Y = np.random.binomial(3, 0.1, (100, 500))

    kwargs = {'name' : 'mydata',
            'reference' : 'made up',
            'category' : 'toy interaction dataset'
            }

    dset = InteractionDataset(Y, **kwargs)

    dset.dump('test.json', dense=True)
    
    dset2 = InteractionDataset.load('test.json')