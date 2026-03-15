__authors__ = '1747579'
__group__ = '14 potser?'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        """
        EXPLICACIO:
            Teoria: eliminar color i agafar pixels de la img. com a caracteristiques.
            vector de 80x60 pixels - dimensio = 4800
            Per tant, cada imatge 80x60x3 (files,col,canals RGB)
            - Eliminar color: es fer la mitjana de canals
            - Convetim la img. en vector: hem de passar de 80x60 a 4800; P=#n imatges
            * Dividim entre 255 pq en RGB els pixels d'una imatge van de 0->negre, 255->blanc
        """
        
        P = train_data.shape[0]
        # convertim RGB -> escala de grisos
        gray = np.mean(train_data, axis = 3)
        self.train_data = gray.reshape(P, -1) / 255.0

    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        # obtenim el nombre d'imatges de test
        n_tests = test_data.shape[0]
        # passem a grisos (mitjana del canal RGB)
        gray = np.mean(test_data, axis = 3)
        test_vectors = gray.reshape(n_tests, -1) / 255.0
        # calculem la distancia entre cad test i cada train
        distances = cdist(test_vectors, self.train_data)
        # ordenem per distancia
        self.neighbors = np.argsort(distances, axis=1)[:, :k] # retorna el indexs
        
        
    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        predictions = []
         
        for i in range(self.neighbors.shape[0]):
            # obtenim les etiquetes dels k veins d'aquesta imatge
            neighbors_labels = self.labels[self.neighbors[i]]
            
            # busquem l'etiqueta que me es repeteix
            unique, counts = np.unique(neighbors_labels, ret = True)
            predictions.append(unique[np.argmx(counts)])
        return np.array(predictions)
        
        
    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        return self.get_class()
