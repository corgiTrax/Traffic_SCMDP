import numpy as np
from cvxopt import matrix, solvers
import solver.sparse.gsc_mdp as GSC
import copy as cp
import roulette
from config import *

np.set_printoptions(precision = 2, linewidth = 1000, suppress = True, threshold = 'nan')

class SCMDP:
    def __init__(self, T, n, A, trans_suc_rate, reward_vec, cap_vec, x0):
        self.reward_vec = np.array(cp.deepcopy(reward_vec))
        self.cap_vec = np.array(cp.deepcopy(cap_vec))
        self.x0 = np.array(cp.deepcopy(x0))
        # length of planning horizon
        self.T = T
        # number of states
        self.n = n
        # number of actions
        self.A = A

        # construct transition matrix G
        self.trans_suc_rate = trans_suc_rate
        self.G = np.zeros((A, n, n))
        for act in range(A):
            # construct the 1-step transition matrix
            for i in range(n):
                self.G[act, act, i] = self.trans_suc_rate 
            for j in range(n):
                if j != act: self.G[act, j, j] = 1 - self.trans_suc_rate 
                else: self.G[act, j, j] = 1
        # print(self.G)

        # construct reward matrix R (over T-1 horizon)
        self.R = np.zeros((T - 1, n, A))
        # stationary reward at t0
        R0 = np.zeros((n, A))
        for i in range(n):
            for j in range(A):
                R0[i][j] = self.reward_vec[i]
        for i in range(T-1):
            self.R[i,:,:] = cp.deepcopy(R0)
        # terminal reward at time T, a nx1 col vector
        self.RT = np.zeros((n, 1))
        for i in range(n):
            self.RT[i][0] = self.reward_vec[i]
        # print(self.R)
        # print(self.RT)

        # construct density vector
        self.d = self.cap_vec.reshape(len(self.cap_vec),1)
        # print(self.d)

        # construct L matrix
        self.L = np.eye(n)
        
        # initial distribution of the agents 
        self.x0 = self.x0.reshape(len(self.x0), 1)
        # print(self.x0)

        # discount factor
        self.gamma = 1 
        
        # policy matrix
        self.bf_Q = []; self.bf_x = []; self.phi_Q = []; self.phi_x = [] 
        self.un_Q = []; self.un_x = []; self.pro_Q = []; self.pro_x = []
        self.unbf_Q = []; self.unbf_x = [];

    def solve(self):
        [self.un_Q, self.un_x, self.phi_Q, self.phi_x, self.bf_Q, self.bf_x,\
        self.pro_Q, self.pro_x, self.unbf_Q, self.unbf_x]\
        = GSC.mdp_cvxpy(self.G, self.R, self.RT, self.L, self.d, self.x0, self.gamma)
        print("scmdp policy solved")

    def save_to_file(self, path):
        '''save all to .npy files'''
        np.save("policy/un_Q", self.un_Q)
        np.save("policy/un_x", self.un_x)
        np.save("policy/phi_Q", self.phi_Q)
        np.save("policy/phi_x", self.phi_x)
        np.save("policy/bf_Q", self.bf_Q)
        np.save("policy/bf_x", self.bf_x)
        np.save("policy/pro_Q", self.pro_Q)
        np.save("policy/pro_x", self.pro_x)
        np.save("policy/unbf_Q", self.unbf_Q)
        np.save("policy/unbf_x", self.unbf_x)

    def load_from_file(self, path):
        self.un_Q = np.load("policy/un_Q.npy")
        self.un_x = np.load("policy/un_x.npy")
        self.phi_Q = np.load("policy/phi_Q.npy")
        self.phi_x = np.load("policy/phi_x.npy")
        self.bf_Q = np.load("policy/bf_Q.npy")
        self.bf_x = np.load("policy/bf_x.npy")
        self.pro_Q = np.load("policy/pro_Q.npy")
        self.pro_x = np.load("policy/pro_x.npy")
        self.unbf_Q = np.load("policy/unbf_Q.npy")
        self.unbf_x = np.load("policy/unbf_x.npy")

    def choose_act(self, state, T, alg):
        if alg == UNC:
            policy = self.un_Q[T][state]
        elif alg == SCPHI:
            policy = self.phi_Q[T][state]
        elif alg == SCPRO:
            # note that projection algorithm is stationary
            policy = self.pro_Q[0][state]
        elif alg == SCBF:
            policy = self.bf_Q[T][state]
        elif alg == SCUBF:
            policy = self.unbf_Q[T][state]
        roulette_selector = roulette.Roulette(policy)
        action = roulette_selector.select()
        # print("Action selected:", action)
        return action

if __name__ == "__main__":
    # initialize scmdp solver
    SCMDP_SELECTOR = SCMDP(T = NUM_EPISODE, n = NUM_STATE, A = NUM_STATE,\
    trans_suc_rate = TRANS_SUC_RATE, reward_vec = REWARD, cap_vec = CAP_DENSITY, x0 = INIT_DENSITY)
    # solve for policy matrix
    SCMDP_SELECTOR.solve()
    SCMDP_SELECTOR.save_to_file(POLICY_PATH)
