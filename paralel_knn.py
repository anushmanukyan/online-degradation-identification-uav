import sys
import collections
import itertools

import numpy as np

from scipy.stats import mode

from multiprocessing import Pool
#from multiprocessing.dummy import Pool as ThreadPool 


from dtw import dtw_distance

import copyreg
import types

def _reduce_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)
copyreg.pickle(types.MethodType, _reduce_method)


class KnnDtw(object):
    
    def __init__(self, k_neighbours=5, max_warping_window=10000):
        self.k_neighbours       = k_neighbours
        self.max_warping_window = max_warping_window
    
    # Public Methods

    def fit(self, x_training_data, x_labels):        
        self.x_training_data = x_training_data
        self.x_labels = x_labels
        
    def predict(self, x):

        p = Pool(3)
        distance_matrix = []
        distance_matrix = self._distance_matrix(x, self.x_training_data)

        jobs = [ (x, [flight]) for flight in self.x_training_data ]
        '''
        for job in jobs:
            distance_matrix = self._distance_matrix(job[0], job[1])
        '''
        parallel_dist = p.map(self._map_single_distance_matrix, jobs)

        #print(parallel_dist)
        #print(type(parallel_dist))        
        
        
        distance_matrix = np.array([parallel_dist])

        #print(distance_matrix)
        #print(type(distance_matrix))        
        
        # Returns only the last k neighbours
        knn_indices = distance_matrix.argsort()[:, :self.k_neighbours]

        # Retrieve the k nearest labels with the indices
        knn_labels = self.x_labels[knn_indices]
        
        # Compute labels and probabilities using the mode (majority vote) ????
        mode_data           = mode(knn_labels, axis=1)

        result_label        = mode_data[0]
        result_probability  = mode_data[1] / self.k_neighbours

        # Return tuple. Ravel is a numpy function that flattens an array.
        # Doc: http://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.ravel.html.
        return result_label.ravel(), result_probability.ravel()

    def _map_single_distance_matrix(self, job_tuple):
        dtw_result = self._distance_matrix(job_tuple[0], job_tuple[1], True)[0]
        return dtw_result

    def _distance_matrix(self, x, y, show_progress=True):
        count = 0
        
        x_shape = np.shape(x)
        y_shape = np.shape(y)

        distance_matrix         = np.zeros((x_shape[0], y_shape[0])) 
        distance_matrix_size    = x_shape[0] * y_shape[0]
        

        for i in range(0, x_shape[0]):
            for j in range(0, y_shape[0]):
                # Compute DTW
                distance_matrix[i, j] = dtw_distance(x[i], y[j], self.max_warping_window)
                
                # Update progress
                count += 1
                if show_progress:
                    self._show_progress(distance_matrix_size, count)


        #print '\r\n'
        return distance_matrix[0]
    
    def _show_progress(self, n, i):
        print ('\r%d/%d %f %%' % (i,n, (float(i)/float(n))*100.0),)
        sys.stdout.flush()


