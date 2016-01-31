
import numpy as np
from cvxopt import matrix, solvers
import gsc_mdp as GSC
T=50
m=2
n=2
A=3
G=np.zeros((A,n,n))
G[0,:,:]=np.array([ [0,1], [1,0] ])
G[1,:,:]=np.array([[1,0],[0,1]])
G[2,:,:]=np.array([[0.5, 0.5],[0.5,0.5]])

R=np.zeros((T-1,n,A))
R0=np.array([[1,1,1],[10,10,10]])
for i in range(T-1):
    R[i,:,:]=R0
RT=np.array([[1],[30]])

# L=np.eye(n)
# d=np.array([[1],[0.4]]
L= np.zeros((m,n))
d= np.zeros((m,1))
L[0,0]=1
L[1,1]=5
d[0,0]=1
d[1,0]=1
x0= np.array([[1],[0]])
gamma=0.99

[un_Q, un_x, phi_Q, phi_x, bf_Q, bf_x ]=GSC.mdp(G, R, RT, L, d, x0, gamma)
print(np.shape(phi_x))
res_un= np.dot(d,np.ones((1,T)))-np.dot(L,un_x)
res_phi= np.dot(d,np.ones((1,T)))-np.dot(L,phi_x)
res_bf=np.dot(d,np.ones((1,T)))-np.dot(L,bf_x)
print(np.amin(res_un))
print(np.amin(res_phi))
print(np.amin(res_bf))
# print(np.dot(L,un_x))
# print(np.dot(L,phi_x))
# print(np.dot(L,bf_x))
