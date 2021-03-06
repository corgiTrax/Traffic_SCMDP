import numpy as np
from config import *
import world
import astar

class Car:
    def __init__(self, identity, start, dest, world, car_type):   
        self.identity = identity
        self.pos = cp.deepcopy(start)
        self.dest = cp.deepcopy(dest)
        self.car_type = car_type
        self.cap = CAP_CAR[self.car_type]
        # the world this agent lives in
        self.world = world
        # update road block
        self.world.world_map[self.pos[ROW]][self.pos[COL]].add_car(self.dest, self.cap)
        self.arrived = False
        self.act = STAY

    def check_arrived(self):
        '''return true if car is already at destination'''
        arrived =  self.pos[ROW] == self.dest[ROW] and self.pos[COL] == self.dest[COL]
        if arrived and self.arrived == False:
            self.arrived = True
        return arrived

    def move(self, action):
        if (self.world.success_move(self.pos, action)):
            self.world.world_map[self.pos[ROW]][self.pos[COL]].rm_car(self.dest, self.cap)
            if (action == UP):
                self.pos[ROW] -=1
            elif (action == DOWN): 
                self.pos[ROW] +=1
            elif (action == LEFT):
                self.pos[COL] -=1
            elif (action == RIGHT):
                self.pos[COL] +=1
            # if action is STAY nothing happens
            self.world.world_map[self.pos[ROW]][self.pos[COL]].add_car(self.dest, self.cap)
    
    def greedy_act(self):
        '''choose the action and move; shortest path with random tie-break'''
        '''TBD, car can get stuck now'''
        dists = []
        for action in ACTIONS:
            dists.append(self.world.dist_act(self.pos, action, self.dest))
        best = [0]
        for i in range(len(ACTIONS)):
            if dists[i] < dists[best[0]]: best[:] = [i]
            elif dists[i] == dists[best[0]]: best.append(i)
        if len(best) == 1: act = best[0]
        elif len(best) > 1: act = random.choice(best)
        self.act = act

    def astar_act(self):
        '''an astar (time-based) agent'''
        path = astar.astar_path(self.world.get_astar_map(), (self.pos[ROW], self.pos[COL]), (self.dest[ROW], self.dest[COL]))
        if len(path) > 0: # has a path
            next_pos = path[-1] 
            if next_pos[ROW] - self.pos[ROW] == -1: act = UP
            elif next_pos[ROW] - self.pos[ROW] == 1: act = DOWN
            elif next_pos[COL] - self.pos[COL] == -1: act = LEFT
            elif next_pos[COL] - self.pos[COL] == 1: act = RIGHT
            self.act = act
        else: self.act = STAY
        # print(path)
    
    def sc_act(self, scmdp_selector, episode, state_dict, alg):
        state_vec = [self.pos[ROW], self.pos[COL], self.dest[ROW], self.dest[COL], self.car_type]
        state =  state_dict.get_num(state_vec)
        self.act = scmdp_selector.choose_act(state, episode, alg)

    def exec_act(self):
        '''execute saved action'''
        self.move(self.act)

    def print_status(self):
        print("{:<6}".format(self.identity)),
        print("({:>2}, {:<2})".format(self.pos[ROW], self.pos[COL])), 
        print("({:>2}, {:<2})".format(self.dest[ROW], self.dest[COL])) 

    def draw(self):
        pass

