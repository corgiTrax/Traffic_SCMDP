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

class Experiment:
    def __init__(self, alg, data_file, vis = True):
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
        
        self.state_dict = state.StateDict(self.test_world) 
        if self.alg == SCMDPBF:
            self.scmdp_selector = scmdp.SCMDP(world_ = self.test_world, sdic_ = self.state_dict, T = NUM_EPISODE, m = self.test_world.num_road, A = len(ACTIONS), trans_suc_rate = TRANS_SUC_RATE)
            self.scmdp_selector.load_from_file()

    def record(self):
        '''write performance data to file'''
        self.data_file.write(str(episode) + ',')
        self.data_file.write(str(self.violation_count))
        self.data_file.write('\n')

    def run(self):
        self.data_file = open(self.data_file_name, 'w')
        for episode in range(NUM_EPISODE - 1): # note this -1
            # visualization
            if self.vis:
                #print("==========================================================")
                #print("{:<6} {:<6} {:<6}".format("CarID", "Position", "Destination"))
                #for car in self.cars:
                #    car.print_status()
                print("Current episode: "), ;print(episode)
                self.test_world.draw()
                if MOUSE == 1: self.test_world.window.getMouse()
            
            car_arrived = 0
            # cars move sequentially
            for car in self.cars:
                if True: #not(car.arrived()):
                    if self.alg == STP:           
                        car.greedy_act() 
                    if self.alg == ASTAR:
                        car.astar_act()
                    if self.alg == SCMDPBF:
                        car.scmdpbf_act(self.scmdp_selector, episode, self.state_dict)
                # else:
                    if car.arrived(): car_arrived += 1

            print("Car Arrived at Destinations:"), ;print(car_arrived)

MOUSE = int(sys.argv[1])                   
new_exp = Experiment(alg = ALG, vis = False, data_file = "data/temp")
new_exp.run()

raw_input("Please Press Enter to Exit")
        
