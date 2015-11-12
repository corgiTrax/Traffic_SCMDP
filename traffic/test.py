from scipy.sparse import csr_matrix, lil_matrix
from scipy import sparse
import numpy as np
A = lil_matrix(np.eye(5))
A[1,:] = [1,2,3,4,5]
(row, col) = np.shape(A)
for i in range(row):
    print(A[i,0])
