import numpy as np
from cvxopt import matrix, solvers
import solver.sparse.gsc_mdp as GSC
import copy as cp
import roulette
from tempfile import TemporaryFile
from scipy import sparse
import time

from config import *
import world
import car
import state
import sys
import os

np.set_printoptions(linewidth = 1000, precision = 2, suppress = True, threshold = 'nan')

def print_matrix(matrix):
    (row,col) = numpy.shape(matrix)
    print("{:<4}".format(' ')), # print head
    for j in range(col): # print col numbers
        print("{:<4}".format(j)),
    print('\n'),
    for i in range(row):
        print("{:<4}".format(i)), # print row numbers
        for j in range(col):
            print("{:<4}".format(matrix[i,j])),
        print('\n'),

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
        
        print("Construction Start")
        start_time = time.time()
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
        self.gamma = GAMMA
        print("Time: ", time.time() - start_time)

        # policy matrix
        self.bf_Q = []; self.bf_x = []; self.phi_Q = []; self.phi_x = [] 
        self.un_Q = []; self.un_x = []; self.pro_Q = []; self.pro_x = []
        self.unbf_Q = []; self.unbf_x = [];

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
                    # if state.same_loc(loc_j, [1,2]): print(act, "at: ", result_loc)
                    for i in range(self.n):
                        state_i = self.sdic.get_state(i)
                        loc_i = [state_i[0], state_i[1]]
                        if state.same_loc(result_loc, loc_i) \
                        and state_i[2] == state_j[2] and state_i[3] == state_j[3] and state_i[4] == state_j[4]:
                            G_act[i][j] = 1
                self.G[act,:,:] = cp.deepcopy(G_act)
#        print_matrix(self.G[0,:,:])
#        print(np.shape(self.G))

    def construct_RT(self):
        ''' n x 1'''
        self.RT = np.zeros((self.n,1)) 
        for i in range(self.n):
            state_vec = self.sdic.get_state(i)
            # if the current position is equal to destination, give a reward
            if state.same_loc([state_vec[0], state_vec[1]], [state_vec[2], state_vec[3]]):
                self.RT[i,0] = REWARD
            # if the current position is equal to start, give a cost
            else:
                dest = [state_vec[2], state_vec[3]]
                start = START[DESTINATION.index(dest)]
                if state.same_loc([state_vec[0], state_vec[1]], [start[0], start[1]]):
                    self.RT[i,0] = COST
                else:
                    self.RT[i,0] = 0.5
#        print_matrix(self.RT)

    def construct_R(self):
        ''' (T-1) x n x A'''
        self.R = np.zeros((self.T-1, self.n, self.A))
        R0 = np.zeros((self.n, self.A))
        for a in range(self.A):
            R0[:,a] = cp.deepcopy(self.RT[:,0])
        for t in range(self.T-1):
            self.R[t,:,:] = cp.deepcopy(R0)
#        print_matrix(self.R[-1])

    def construct_L(self):
        ''' m x n'''
        #self.L = np.zeros((self.m, self.n))
        # note: if we have more than or less than two types of car this need to change
        I_SMALL = CAP_SMALL * np.eye(self.m) # small car
        I_BIG = CAP_BIG * np.eye(self.m) # big car
        self.L = I_SMALL
        for i in range(len(DESTINATION) - 1):
            self.L = np.append(self.L, I_SMALL, axis = 1)
        for i in range(len(DESTINATION)):
            self.L = np.append(self.L, I_BIG, axis = 1)
        # take out the capacity bounds for start end end locations
        self.L_cst = cp.deepcopy(self.L) # back up constrained L
        if REMOVE_CONSTRAINT:
            pos_count = 0; remove_count = 0
            for i in range(self.world.rows):
                for j in range(self.world.columns):
                    if self.world.world_map[i][j].block_type != OFFROAD:
                        if [i,j] in START or [i,j] in DESTINATION:
                            self.L = np.delete(self.L, (pos_count - remove_count), axis = 0)
                            remove_count += 1
                        pos_count += 1
#        print_matrix(self.L)

    def construct_d(self):
        ''' m x 1'''
        self.d = np.zeros((self.m, 1))
        state_count = 0
        for i in range(self.world.rows):
            for j in range(self.world.columns):
                if self.world.world_map[i][j].block_type != OFFROAD:
                    self.d[state_count, 0] = 1.0 * self.world.world_map[i][j].cap_bound / NUM_CAR
                    state_count += 1
        # take out the capacity bounds for start end end locations
        if REMOVE_CONSTRAINT:
            pos_count = 0; remove_count = 0;
            for i in range(self.world.rows):
                for j in range(self.world.columns):
                    if self.world.world_map[i][j].block_type != OFFROAD:
                        if [i,j] in START or [i,j] in DESTINATION:
                            self.d = np.delete(self.d, (pos_count - remove_count), axis = 0)
                            remove_count += 1
                        pos_count += 1
#        print_matrix(self.d)

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
#        print_matrix(self.x0)

    def solve(self):
#        [self.un_Q, self.un_x, self.phi_Q, self.phi_x, self.bf_Q, self.bf_x, self.pro_Q, self.pro_x, self.unbf_Q, self.unbf_x] = \
#        GSC.mdp_cvxpy(self.G, self.R, self.RT, self.L, self.d, self.x0, self.gamma)
        [self.unbf_Q, self.unbf_x] = \
        GSC.mdp_cvxpy(self.G, self.R, self.RT, self.L, self.d, self.x0, self.gamma)

        print("Scmdp policy solved")

#        print("un_Q", self.un_Q)
#        print("un_x: "); print(self.un_x)
#        print("L*unx: ");print(np.dot(self.L_cst, self.un_x))

#        print("phi_Q", self.phi_Q)
#        print("phi_x: "); print(self.phi_x)
#        print("L*phix: ");print(np.dot(self.L_cst, self.phi_x))

#        print("bf_Q", self.bf_Q)
#        print("bf_x: "); print(self.bf_x)
#        print("L*bfx: ");print(np.dot(self.L_cst, self.bf_x))

#        print("pro_Q", self.pro_Q)
#        print("pro_x: "); print(self.pro_x)
#        print("L*prox: ");print(np.dot(self.L_cst, self.pro_x))

#        print("pro_Q", self.pro_Q)
#        print("unbf_x: "); print(self.unbf_x)
#        print("L*unbfx: ");print(np.dot(self.L_cst, self.unbf_x))

    def save_to_file(self):
        '''save all to .npy files'''
        directory = "policy/world2/" + str(NUM_CAR) + "/" 
        if not os.path.exists(directory):
            os.makedirs(directory)
#        np.save(directory + "un_Q", self.un_Q)
#        np.save(directory + "phi_Q", self.phi_Q)
#        np.save(directory + "bf_Q", self.bf_Q)
#        np.save(directory + "pro_Q", self.pro_Q)
        np.save(directory + "unbf_Q", self.unbf_Q)

    def load_from_file(self):
        directory = "policy/world2/" + str(NUM_CAR) + "/" 
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.un_Q = np.load(directory + "un_Q.npy")
        self.phi_Q = np.load(directory + "phi_Q.npy")
        self.bf_Q = np.load(directory + "bf_Q.npy")
        self.pro_Q = np.load(directory + "pro_Q.npy")
        self.unbf_Q = np.load(directory + "unbf_Q.npy")

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
    # call solve and store resulted matrices
    start_time = time.time()
    test_world = world.World()
    state_dict = state.StateDict(test_world) 
    scmdp_solver = SCMDP(world_ = test_world, sdic_ = state_dict, T = NUM_EPISODE, m = test_world.num_road, A = len(ACTIONS), trans_suc_rate = TRANS_SUC_RATE)
    scmdp_solver.solve()
    scmdp_solver.save_to_file()
    print("Total computation time: "),;print(time.time() - start_time)
