import sys
import numpy as np


def distance(x, y):
        return abs( x - y )

def dtw_distance(timeserie_a, timeserie_b, max_warping_window):

    # Initialising variables with numpy structures
    timeserie_a = np.array(timeserie_a)
    timeserie_b = np.array(timeserie_b)

    M = len(timeserie_a)
    N = len(timeserie_b)

    # Create cost matrix by filling it with very large integers, because the dynamic programming functions
    # uses the min(...) function. Thus, it is better then initializing with 0 or None.
    cost_matrix = sys.maxsize * np.ones( (M, N) )

    # Initialising the first cell
    cost_matrix[0, 0] = distance(timeserie_a[0], timeserie_b[0])

    # Initializing the first row
    for i in range(1, M):
        cost_matrix[i, 0] = cost_matrix[i-1, 0] + distance(timeserie_a[i], timeserie_b[0])

    # Initializing the first column
    for i in range(1, N):
        cost_matrix[0, i] = cost_matrix[0, i-1] + distance(timeserie_a[0], timeserie_b[i])

    # Run trough the reste of the cost_matrix and stay withing the limits of the
    # warping window. Performs the "main dynamic programming function".
    for i in range(1, M):
        from_max_warping_window = max(1, i - max_warping_window)
        to_max_warping_window   = min(N, i + max_warping_window)

        for j in range(from_max_warping_window, to_max_warping_window):
            choice_1 = cost_matrix[i - 1, j - 1]
            choice_2 = cost_matrix[i - 1, j    ]
            choice_3 = cost_matrix[i    , j - 1]

            cost_min_choices = min(choice_1, choice_2, choice_3)

            cost_matrix[i, j] = cost_min_choices + distance(timeserie_a[i], timeserie_b[j])

    # Return the DTW distance ([-1,-1] return the last item)
    return cost_matrix[-1, -1]




