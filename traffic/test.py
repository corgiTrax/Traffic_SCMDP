from scipy.sparse import coo_matrix, csr_matrix, lil_matrix
from scipy import sparse
import numpy as np
import sys
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
    #A = np.random.rand(dim, dim)
    #B = np.random.rand(dim, dim)
    A = np.eye(dim)
    print(A.shape)
    #C = np.concatenate((A,B), axis = 1)
    #D = lil_matrix(A[1,:,:])
    #print(D.shape)
else:
    A = lil_matrix(np.ones((3,4)))
    B = A.reshape((4,3))
    print(A.toarray())
    print(B.toarray())
    #A = matrix(A.todense())
    #A = matrix(A.toarray())
    #print(A.size)
    #B = lil_matrix((dim, dim))
    #C = lil_matrix((dim, dim))
    #D = lil_matrix((dim, dim))
    #F = np.zeros((dim, dim))
    #E = sparse.hstack([A,B,C,D,F,np.ones((dim, dim))])
    #print(E.shape)
    #A = A.tocoo()
    #B = B.tocoo()
    #C = sparse.hstack([A,B])
    #B[0, 7] = 7
    #A[2, :] = 3 * B
    #print(A.toarray())
    #A = A.tocoo()
    #A = scipy.sparse.hstack([A,A])
    #print(A.shape)
    #A= sparse.rand(dim, dim, density = 0.0)
    #B= sparse.rand(dim, dim, density = 0.0)
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
