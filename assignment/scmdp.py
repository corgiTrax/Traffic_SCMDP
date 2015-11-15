import numpy as np
from cvxopt import matrix, solvers
import scmdp_solver.sparse.gsc_mdp as GSC
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
        # self.bf_Q = []
        # self.bf_x = []
        # self.phi_Q = []
        # self.phi_x = []

    def solve(self):
        [self.phi_Q, self.phi_x, self.bf_Q, self.bf_x] = GSC.mdp(self.G, self.R, self.RT, self.L, self.d, self.x0, self.gamma)
        print("scmdp policy solved")
#        print("phiX: ", self.phi_x)
#        print("bfX: ", self.bf_x)
#        res_un = np.dot(self.d, np.ones((1, self.T))) - np.dot(self.L, un_x)
#        res_phi = np.dot(self.d, np.ones((1, self.T))) - np.dot(self.L, phi_x)
#        res_bf = np.dot(self.d, np.ones((1, self.T))) - np.dot(self.L, bf_x)
#        print(np.amin(res_un))
#        print(np.amin(res_phi))
#        print(np.amin(res_bf))
#        print(np.dot(self.L,un_x))
#        print(np.dot(self.L,phi_x))
#        print(np.dot(self.L,bf_x))

    def save_to_file(self, path):
        '''save un_Q, un_x, phi_Q, phi_x, bf_Q, bf_x to .npy files'''
#        np.save("policy/un_Q", self.un_Q)
#        np.save("policy/un_x", self.un_x)
        np.save(path + "phi_Q", self.phi_Q)
        np.save(path + "phi_x", self.phi_x)
        np.save(path + "bf_Q", self.bf_Q)
        np.save(path + "bf_x", self.bf_x)

    def load_from_file(self, path):
        self.phi_Q = np.load(path + "phi_Q.npy")
        self.phi_x = np.load(path + "phi_x.npy")
        self.bf_Q = np.load(path + "bf_Q.npy")
        self.bf_x = np.load(path + "bf_x.npy")

    def choose_act(self, state, T):
        policy = self.bf_Q[T][state]
        # print("Policy vector", policy)
        roulette_selector = roulette.Roulette(policy)
        action = roulette_selector.select()
        # print("Action selected:", action)
        return action

    def choose_act_phi(self, state, T):
        policy = self.phi_Q[T][state]
        # print("Policy vector", policy)
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
