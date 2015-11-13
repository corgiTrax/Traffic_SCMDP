import numpy as np
from cvxopt import matrix, solvers
import scmdp_solver.gsc_mdp as GSC
import copy as cp
import roulette
from tempfile import TemporaryFile

from config import *
import world
import car
import state

np.set_printoptions(linewidth = 1000, precision = 3, suppress = True, threshold = 'nan')

def print_m(matrix):
    '''visualize a 2d matrix'''
    (row,col) = np.shape(matrix)
    print("   "),
    for j in range(col):
        print(str(j) + ' '),
    print(" ")
    for i in range(row):
        print(i, matrix[i])

def print_part(matrix, row_s, row_e, col_s, col_e):
    print("{:<4}".format(' ')), # print head
    for j in range(col_e - col_s + 1): # print col numbers
        print("{:<4}".format(j)),
    print('\n'),
    for i in range(row_s, row_e + 1):
        print("{:<4}".format(i)), # print row numbers
        for j in range(col_s, col_e + 1):
            print("{:<4}".format(matrix[i,j])),
        print('\n'),

class SCMDP:
    def __init__(self, world_, sdic_, T, A, m, trans_suc_rate):
        self.world = world_
        self.sdic = sdic_ 
        self.T = T  # length of planning horizon
        self.n = self.sdic.n  # number of states
        self.A = A  # number of actions
        self.m = m  # number of constraints
        self.trans_suc_rate = trans_suc_rate # transition success rate       

        # construct transition matrix G
        self.construct_G()  
        # construct reward matrix R (over T-1 horizon)
        self.construct_RT() 
        self.construct_R()
        # construct density vector
        self.construct_d()
        # construct L matrix
        self.construct_L()
        # initial distribution of the agents 
        self.construct_x0()
        # discount factor
        self.gamma = 0.99
        
        # policy matrix
        self.bf_Q = []; self.bf_x = []; self.phi_Q = []; self.phi_x = []; self.un_Q = []; self.un_x = []

    def construct_G(self):
        '''A x n x n'''
        self.G = np.zeros((self.A, self.n, self.n))
        for act in range(self.A):
            if act == STAY:
                self.G[act,:,:] = np.eye(self.n) # stay results in an identity matrix
            else:
                # probability from j to i
                G_act = np.zeros((self.n, self.n))
                for j in range(self.n):
                    state_j = self.sdic.get_state(j) # start from this state
                    loc_j = [state_j[0], state_j[1]]
                    result_loc = self.world.move_consq(loc_j, act)
                    for i in range(self.n):
                        state_i = self.sdic.get_state(i)
                        loc_i = [state_i[0], state_i[1]]
                        if state.same_loc(result_loc, loc_i) \
                        and state_i[2] == state_j[2] and state_i[3] == state_j[3] and state_i[4] == state_j[4]:
                            G_act[i][j] = 1
                self.G[act,:,:] = cp.deepcopy(G_act)
        #print_part(self.G[LEFT], 0, 167, 167, 167)
        #print(np.shape(self.G))

    def construct_RT(self):
        ''' n x 1'''
        self.RT = np.zeros((self.n,1)) 
        for i in range(self.n):
            state_vec = self.sdic.get_state(i)
            # if the current position is equal to destination
            if state.same_loc([state_vec[0], state_vec[1]], [state_vec[2], state_vec[3]]):
                self.RT[i,0] = REWARD
        #print_m(self.RT)

    def construct_R(self):
        ''' (T-1) x n x A'''
        self.R = np.zeros((self.T-1, self.n, self.A))
        R0 = np.zeros((self.n, self.A))
        for a in range(self.A):
            R0[:,a] = cp.deepcopy(self.RT[:,0])
        for t in range(self.T-1):
            self.R[t,:,:] = cp.deepcopy(R0)
        # print_m(self.R[1])

    def construct_L(self):
        ''' m x n'''
        #self.L = np.zeros((self.m, self.n))
        # note: if we have more than two types of car this need to change
        I_SMALL = CAP_SMALL * np.eye(self.m) # small car
        I_BIG = CAP_BIG * np.eye(self.m) # big car
        self.L = I_SMALL
        for i in range(len(DESTINATION) - 1):
            self.L = np.append(self.L, I_SMALL, axis = 1)
        for i in range(len(DESTINATION)):
            self.L = np.append(self.L, I_BIG, axis = 1)
        # print_m(self.L)

    def construct_d(self):
        ''' m x 1'''
        self.d = np.zeros((self.m, 1))
        state_count = 0
        for i in range(self.world.rows):
            for j in range(self.world.columns):
                if self.world.world_map[i][j].block_type != OFFROAD:
                    self.d[state_count, 0] = 1.0 * self.world.world_map[i][j].cap_bound / NUM_CAR
                    state_count += 1
        # print_m(self.d)

    def construct_x0(self):
        ''' n x 1, assume cars are distributed equally in start '''
        self.x0 = np.zeros((self.n, 1))
        for i in range(self.n):
            state_vec = self.sdic.get_state(i)
            start_pos = [state_vec[0], state_vec[1]]
            if not(start_pos in START):
                continue
            else:
                des_pos = [state_vec[2], state_vec[3]]
                if state.same_loc(DESTINATION[START.index(start_pos)], des_pos):
                    self.x0[i, 0] = INIT_DENSITY_CORNER
        # print_m(self.x0)

    def solve(self):
        [self.un_Q, self.un_x, self.phi_Q, self.phi_x, self.bf_Q, self.bf_x] = GSC.mdp(self.G, self.R, self.RT, self.L, self.d, self.x0, self.gamma)
        print("scmdp policy solved")
#        print(self.bf_Q)
        print(np.dot(self.L, self.bf_x))
#        res_un = np.dot(self.d, np.ones((1, self.T))) - np.dot(self.L, un_x)
#        res_phi = np.dot(self.d, np.ones((1, self.T))) - np.dot(self.L, phi_x)
#        res_bf = np.dot(self.d, np.ones((1, self.T))) - np.dot(self.L, bf_x)
#        print(np.amin(res_un))
#        print(np.amin(res_phi))
#        print(np.amin(res_bf))
#        print(np.dot(self.L,un_x))
#        print(np.dot(self.L,phi_x))
#        print(np.dot(self.L,bf_x))

    def save_to_file(self):
        '''save un_Q, un_x, phi_Q, phi_x, bf_Q, bf_x to .npy files'''
        np.save("policy/un_Q", self.un_Q)
        np.save("policy/un_x", self.un_x)
        np.save("policy/phi_Q", self.phi_Q)
        np.save("policy/phi_x", self.phi_x)
        np.save("policy/bf_Q", self.bf_Q)
        np.save("policy/bf_x", self.bf_x)

    def load_from_file(self):
        self.un_Q = np.load("policy/un_Q.npy")
        self.un_x = np.load("policy/un_x.npy")
        self.phi_Q = np.load("policy/phi_Q.npy")
        self.phi_x = np.load("policy/phi_x.npy")
        self.bf_Q = np.load("policy/bf_Q.npy")
        self.bf_x = np.load("policy/bf_x.npy")

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
    # call solve and store resulted matrices
    test_world = world.World()
    state_dict = state.StateDict(test_world) 
    scmdp_solver = SCMDP(world_ = test_world, sdic_ = state_dict, T = 5, m = test_world.num_road, A = len(ACTIONS), trans_suc_rate = TRANS_SUC_RATE)
    scmdp_solver.solve()
    scmdp_solver.save_to_file()
