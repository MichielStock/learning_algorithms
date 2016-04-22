"""
Created on Fri Apr 22 2016
Last update: -

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
        self.data_type = str(self.Y.dtype)

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

        if self.u_objects is None:
            u_objects = range(self.n_rows)
        else:
            u_objects = self.u_objects
        if self.v_objects is None:
            v_objects = range(self.n_cols)
        else:
            v_objects = self.v_objects

        if dense == True:
            graph = {sp_row : list(self.Y[i])
            for i, sp_row in enumerate(u_objects)}
        else:
            graph = {sp_row : [(sp_col, self.Y[i, j])
            for j, sp_col in enumerate(v_objects)
            if self.Y[i, j] != 0]
            for i, sp_row in enumerate(u_objects)}
        return graph

if __name__ == '__main__':
    import numpy as np

    Y = np.random.binomial(3, 0.1, (100, 500))

    kwargs = {'name' : 'mydata',
            'reference' : 'made up',
            'category' : 'toy interaction dataset'
            }

    dset = InteractionDataset(Y, **kwargs)
