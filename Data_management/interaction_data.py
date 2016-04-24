"""
Created on Fri Apr 22 2016
Last update: Sun Apr 24 2016

@author: Michiel Stock
michielfmstock@gmail.com

Module for storing interaction datasets

Each interaction dataset has following properties:
    - interaction (adjacency) matrix
    - name
    - general metadata
    - metadata rows
    - metadata columns
    - version number
"""

import json
import numpy as np

def dense_graph_to_matrix(shape, graph, dtype=float):
    """
    Turns dense graph into a numpy matrix
    """
    Y = np.zeros(shape, dtype=dtype)
    for i, vals in graph.items():
        # keys in json are str
        Y[int(i), :] = np.array(vals)[:]
    return Y

def sparse_graph_to_matrix(shape, graph, dtype=float):
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
        - interaction (adjacency) matrix
        - name
        - general metadata
        - metadata rows
        - metadata columns
        - version number

    Here we use the convention that 'u' refers to objects of the rows and 'v'
    to objects of the columns.
    """
    def __init__(self, interaction_matrix, name=None, metadata=None,
                     u_metadata=None, v_metadata=None):
        """
        Construct an interaction dataset with a given interaction matrix and
        (optionally) a name and general, rows and columns metadata
        """
        # interaction matrix
        self.Y = interaction_matrix  # e.g. a protein-ligand interaction binding
        # matrix

        # metadata
        self.name = name
        self.metadata = metadata

        # information about the rows
        self.u_metadata = u_metadata

        # information about the columns
        self.v_metadata = v_metadata

        # propperties about the dataset
        self.n_rows, self.n_cols = self.Y.shape
        self.density = np.sum(Y != 0) / self.n_rows * self.n_cols

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
                'name' : self.name,
                'metadata' : self.metadata,
                'u_metadata' : self.u_metadata,
                'v_metadata' : self.v_metadata,
                'interactions' : self.make_graph(dense),
                'n_rows' : self.n_rows,
                'n_cols' : self.n_cols,
                'density' : self.density,
                'storage' : 'dense' if dense else 'sparse'
                }
        json.dump(data, fh, indent=indent, separators=(',', ':'))

    @classmethod
    def load_csv(self, filename, kwargs):
        """
        Load a dataset from csv format
        """
        Y = np.genfromtxt(filename)
        return self(Y, **kwargs)

    @classmethod
    def load_json(self, filename):
        """
        Load a dataset from JSON format
        """
        fh = open(filename, 'r')
        data = json.load(fh)
        assert data['version'] == 1
        # load interaction matrix
        shape = (data['n_rows'], data['n_cols'])
        graph = data['interactions']
        if data['storage'] == 'dense':
            Y = dense_graph_to_matrix(shape, graph)
        elif data['storage'] == 'sparse':
            Y = sparse_graph_to_matrix(shape, graph)
        else:
            raise KeyError('storage format is either \'dense\' or \'sparse\'')
        return self(Y, name=data['name'],
                    metadata=data['metadata'],
                     u_metadata=data['u_metadata'],
                    v_metadata=data['v_metadata'])

if __name__ == '__main__':

    Y = np.random.binomial(3, 0.1, (100, 500))

    name = 'simulated dataset'
    metadata = 'no metadata'
    u_metadata = map(str, range(100))
    v_metadata = map(str, range(500))

    dset = InteractionDataset(Y,
                              name=name,
                              metadata=metadata,
                              v_metadata=v_metadata,
                              u_metadata=u_metadata)

    dset.dump('test.json', dense=False, indent=2)

    dset2 = InteractionDataset.load_json('test.json')
