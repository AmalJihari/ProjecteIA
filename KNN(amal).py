__authors__ = ['1747579', '1744604', '1744896']
__group__ = '17'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist


class KNN:
    def __init__(self, train_data, labels, options=None):
            # Guardamos las opciones primero de todo para que _init_train las pueda leer
            if options is None:
                self.options = {}
            else:
                self.options = options
                
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
        train_data = train_data.astype(float)
        P = train_data.shape[0]
        
        metode = getattr(self, 'options', {}).get('knn_features', 'pixel').lower()

      
        if metode == 'pixel':
            self.train_data = train_data.reshape(P, -1)
            
        #Millora 2: 
        elif metode == 'reduccio':
            #es selecciona un pixel si i un no, per files i columnes
            imatges_reduides = train_data[:, ::2, ::2]
            self.train_data = imatges_reduides.reshape(P, -1)
            
       
        #Millora 3:
        elif metode == 'descriptors':
            #calcul lluminositat:
            mitjanes = np.mean(train_data, axis=(1, 2, 3)).reshape(P, 1)
            variancies = np.var(train_data, axis=(1, 2, 3)).reshape(P, 1)
            #calcul variança:
            self.train_data = np.hstack((mitjanes, variancies))
            
        else: 
            self.train_data = train_data.reshape(P, -1)
       
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
        test_data = np.array(test_data).astype(float)
        num_tests = test_data.shape[0]
        
        # Obtenim el mètode de les opcions per aplicar la mateixa transformació al test
        metode = getattr(self, 'options', {}).get('knn_features', 'pixel').lower()
        
        if metode == 'pixel':
            test_transformat = test_data.reshape(num_tests, -1)
            
            
        #2: Recorre els pixels de la imatge de test de 2 en 2
        elif metode == 'reduccio':
            imatges_test_reduides = test_data[:, ::2, ::2]
            test_transformat = imatges_test_reduides.reshape(num_tests, -1)
            
        elif metode == 'descriptors':
            mitjanes_test = np.mean(test_data, axis=(1, 2, 3)).reshape(num_tests, 1)
            variancies_test = np.var(test_data, axis=(1, 2, 3)).reshape(num_tests, 1)
            test_transformat = np.hstack((mitjanes_test, variancies_test))
            
        else:
            test_transformat = test_data.reshape(num_tests, -1)

        # 1: Tipus de distància, segons com configurem les opcions
        #Per defecte sera 'euclidean'
        #Segons com configurem les opcions, Manhhatan sera 'cityblock'
        tipus_distancia = getattr(self, 'options', {}).get('knn_dist', 'euclidean').lower()
        distancies = cdist(test_transformat, self.train_data, tipus_distancia)
        
        indexs_mes_proxims = np.argsort(distancies, axis=1)[:, :k]
        self.neighbors = self.labels[indexs_mes_proxims]
        
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
            # obtenim les etiquetes dels k veins d'aquesta imatge (com que es una matriu, anem fila per fila).
            neighbors_labels = self.neighbors[i]
            
            # busquem l'etiqueta que me es repeteix
            # values:quins noms hi ha, counts:quants cops es repeteix
            values, counts = np.unique(neighbors_labels, return_counts = True)
            #busquem quins elements tenen el numero més alt
            candidates = values[counts == np.max(counts)]

            # escollim el que estugui més a prop (desempat)
            for label in neighbors_labels:
                if label in candidates:
                    predictions.append(label)
                    break
        return np.array(predictions)
        
        
    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """

        self.get_k_neighbours(test_data, k)
        # un cop tenim els veins guardats, mirem quin més es repeteix.
        return self.get_class()
