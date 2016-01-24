"""
2D grid world simulation of traffic control
Experiment file
"""
from config import *
import world
import car
import scmdp
import state
import sys
import os

class Experiment:
    def __init__(self, alg, data_file, vis = True, mouse = 0):
        # initialize the world
        self.test_world = world.World()
        # initialize car
        self.cars = []
         
        # assign cars to their start locations
        num_car_sec = NUM_CAR / (len(CAR_TYPE) * len(START))
        car_count = 0
        for pos in START:
            for cartype in CAR_TYPE:
                for i in range(num_car_sec):
                    new_car = car.Car(identity = car_count, start = pos, dest = DESTINATION[START.index(pos)], \
                    world = self.test_world, car_type = cartype)
                    self.cars.append(new_car)
                    car_count += 1
        
        self.alg = alg

        self.data_file_name = data_file
        self.vis = vis
        if self.vis:
            self.test_world.draw(isNew = True)
        self.mouse = mouse

        self.state_dict = state.StateDict(self.test_world) 
        if self.alg in [UNC, SCPHI, SCPRO, SCBF, SCUBF]:
            self.scmdp_selector = scmdp.SCMDP(world_ = self.test_world, sdic_ = self.state_dict, T = NUM_EPISODE, m = self.test_world.num_road, A = len(ACTIONS), trans_suc_rate = TRANS_SUC_RATE)
            self.scmdp_selector.load_from_file()

    def record(self, episode):
        '''write performance data to file'''
        self.data_file.write(str(episode) + ',')
        self.data_file.write(str(self.cap_arrived))
        self.data_file.write('\n')

    def run(self):
        self.data_file = open(self.data_file_name, 'w')
        # count arrived cars and capacity
        self.car_arrived = 0; self.cap_arrived = 0;
        for episode in range(NUM_EPISODE - 1): # note this -1
            # visualization
            if self.vis:
                #print("==========================================================")
                #print("{:<6} {:<6} {:<6}".format("CarID", "Position", "Destination"))
                #for car in self.cars:
                #    car.print_status()
                print("Current episode: "), ;print(episode)
                self.test_world.draw()
                if self.mouse == 1: self.test_world.window.getMouse()
            
            # cars make decision simutaneously except ASTAR
            for car in self.cars:
                if not(car.arrived):
                    if self.alg == STP:           
                        car.greedy_act() 
                    elif self.alg == ASTAR:
                        car.astar_act()
                    else: 
                        # heuristic to improve efficiency of SCMDP algorithms
                        if TOTAL_CAP - self.cap_arrived <= self.test_world.min_cap and SCMDP_STP == True:
                            # print("Switched to STP algorithm")
                            car.greedy_act()
                        else:
                            car.sc_act(self.scmdp_selector, episode, self.state_dict, self.alg)
                    # ASTAR cars move sequentially
                    if self.alg == ASTAR: 
                        car.exec_act()
                        if car.check_arrived(): 
                            self.car_arrived += 1
                            self.cap_arrived += car.cap
            
            # execute the move
            if self.alg != ASTAR:
                for car in self.cars:
                    if not(car.arrived):
                        car.exec_act()
                        if car.check_arrived(): 
                            self.car_arrived += 1
                            self.cap_arrived += car.cap
            
            # recording 
            self.record(episode)
            print("Capacities Arrived at Destinations:"), ;print(self.cap_arrived)
            
        # visualization of last step
        if self.vis:
            #print("==========================================================")
            #print("{:<6} {:<6} {:<6}".format("CarID", "Position", "Destination"))
            #for car in self.cars:
            #    car.print_status()
            print("Current episode: "), ;print(episode + 1)
            self.test_world.draw()
            if self.mouse == 1: self.test_world.window.getMouse()
            print("Capacities Arrived at Destinations:"), ;print(self.cap_arrived)

def main():
    for algorithm in ALGS:
        for trial in range(NUM_EXP):
            directory = "data/" + str(NUM_CAR) + "/" 
            if not os.path.exists(directory):
                os.makedirs(directory)
            filename = directory + str(ALGS_NAME[algorithm]) + str(trial) + ".data"
            new_exp = Experiment(alg = algorithm, vis = False, data_file = filename, mouse = 0) 
            new_exp.run()

def test():
    new_exp = Experiment(alg = ALG, vis = False, data_file = "data/temp", mouse = int(sys.argv[1])) 
    new_exp.run()

#main()
test()

raw_input("Please Press Enter to Exit")
        
