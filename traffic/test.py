from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from scipy import sparse
import numpy as np
import sys
import scipy
from cvxopt import solvers, matrix, spmatrix, mul

dim = 40000
sp = sys.argv[1]

def scipy_sparse_to_spmatrix(A):
    coo = A.tocoo()
    SP = spmatrix(coo.data, coo.row.tolist(), coo.col.tolist(), coo.shape, tc='d')
    return SP
 
def spmatrix_sparse_to_scipy(A):
    data = np.array(A.V).squeeze()
    rows = np.array(A.I).squeeze()
    cols = np.array(A.J).squeeze()
    return sparse.coo_matrix( (data, (rows, cols)) )



if int(sp) == 0:
    A = np.random.rand(dim, dim)
    B = np.random.rand(dim, dim)
    C = np.concatenate((A,B), axis = 1)
else:
    A = lil_matrix(sparse.eye(dim))
    A[2, 3] = 5678
    A = A.tocoo()
    A = scipy.sparse.hstack([A,A])
    print(A.shape)
    #A= scipy.sparse.rand(dim, dim, density = 0.001)
    #B= scipy.sparse.rand(dim, dim, density = 0.001)
    #A = A.tocoo()
    #print(scipy.sparse.isspmatrix_coo(A))    
    #C = scipy.sparse.hstack([A, B])
    #print(np.shape(C))
#    Aeq= scipy.sparse.rand(dim, dim, density = 0.1)
#    beq= scipy.sparse.rand(dim, 1, density = 0.01)
#    Aineq= scipy.sparse.rand(dim, dim, density = 0.1)
#    bineq= scipy.sparse.rand(dim, 1, density = 0.01)
#    Aeq= matrix(np.random.rand(dim, dim))
#    beq= matrix(np.random.rand(dim, 1))
#    Aineq= matrix(np.random.rand(dim, dim))
#    bineq= matrix(np.random.rand(dim, 1))
#    print(Aineq.shape)
#    print(Aeq.shape)

#    c = matrix(np.random.rand(dim,1))
#    Aeq = scipy_sparse_to_spmatrix(Aeq)
#    beq = scipy_sparse_to_spmatrix(beq)
#    Aineq = scipy_sparse_to_spmatrix(Aineq)

#    A = coo_matrix((5,6))
#   A = scipy.sparse.hstack([A,A])
#    print(A.toarray())

#    bineq = scipy_sparse_to_spmatrix(bineq)
#    sol=solvers.lp(c,Aineq,bineq,Aeq,beq)
#    print(sol)
#A = lil_matrix(np.eye(5))
#A[1,:] = [1,2,3,4,5]
#(row, col) = np.shape(A)
#for i in range(row):
#    print(A[i,0])
