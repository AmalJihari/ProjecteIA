__authors__ = ['1747579', '1744604', '1744896']
__group__ = '17'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        f=np.asarray(X, dtype = float)
    
        if f.ndim> 2:
            self.X=f.reshape(-1, X.shape[-1])
        else:
            self.X=f

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
       

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        if self.options['km_init'].lower() == 'first':
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])
        else:
            self.centroids = np.random.rand(self.K, self.X.shape[1])
            self.old_centroids = np.random.rand(self.K, self.X.shape[1])
        """
        self.centroids=np.zeros((self.K, self.X.shape[1]))
        self.old_centroids=np.zeros((self.K, self.X.shape[1]))
       
        if self.options['km_init'].lower()=='first':
            unique_points=[]
            for p in self.X:
                duplicat=any(np.array_equal(p,u) for u in unique_points)
                if not duplicat:
                    unique_points.append(p)
               
                if len(unique_points)==self.K:
                    break
           
            self.centroids=np.array(unique_points)
           
        elif self.options['km_init'].lower()=='random':
            indices=np.random.choice(self.X.shape[0], self.K, replace=False)
            self.centroids=self.X[indices].copy()
           
        #Opcio CUSTOM:
        else:
            for i in range(self.K):
                if self.K>1:
                    val=(1*255)/(self.K-1)
                else:
                    val=127
               
                self.centroids[i]=np.full(self.X.shape[1], val)
       
        self.old_centroids=np.zeros_like(self.centroids)

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        d = distance(self.X, self.centroids)
        #busquem el q té el valor minim (horitzontalment)
        self.labels = np.argmin(d, axis=1)
        
    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self.old_centroids=self.centroids.copy()
       
        for k in range (self.K):
            e=self.labels==k
            punts=self.X[e]
           
            if len(punts)>0:
                self.centroids[k]=np.mean(punts, axis=0)

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        diff = np.linalg.norm(self.centroids - self.old_centroids)
        return (diff <= self.options['tolerance'])

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        self._init_centroids()
        self.num_iter = 0
        
        while self.num_iter < self.options['max_iter']:
            self.get_labels()
            self.get_centroids()
            
            self.num_iter +=1
            # comprovem convergencia
            if self.converges():
                break

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        # FORMULA
        centroids_assigned = self.centroids[self.labels] #NxD
        #diferencia punt - centroide
        diff = self.X - centroids_assigned
        # distancia quadrada
        sq_dist = np.sum(diff**2, axis=1)
        # mitjana
        self.WCD = np.mean(sq_dist)
        return self.WCD

    def find_bestK(self, max_K):
        """
         sets the best k analysing the results up to 'max_K' clusters
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        wcds = []
        
        for k in range(2, max_K+1):
            km = KMeans(self.X, K=k, options = self.options)
            km.fit()
            wcds.append(km.withinClassDistance())
                
        for i in range(1, len(wcds)):
            dec = 100 * (wcds[i] / wcds[i-1])
            if (100 - dec) < 20:
                self.K = i + 1
                return self.K
    
        self.K = max_K
        return self.K
    

def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################

    #matriu buida
    X_sq = np.sum(X**2, axis = 1, keepdims= True) # Nx1
    C_sq = np.sum(C**2, axis = 1) # K
    creu = X @ C.T # NxK
    dist = np.sqrt(X_sq + C_sq -2 * creu)
    return dist
    
def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    return [utils.colors[np.argmax(p)] for p in utils.get_color_prob(centroids)]
