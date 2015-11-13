"""
2D grid world simulation of traffic control
Experiment file
"""
from config import *
import world
import car

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

    def record(self):
        '''write performance data to file'''
        self.data_file.write(str(episode) + ',')
        self.data_file.write(str(self.violation_count))
        self.data_file.write('\n')

    def run(self):
        self.data_file = open(self.data_file_name, 'w')
        while (True):
            # visualization
            if self.vis:
                #print("==========================================================")
                #print("{:<6} {:<6} {:<6}".format("CarID", "Position", "Destination"))
                #for car in self.cars:
                #    car.print_status()
                self.test_world.draw()
                self.test_world.window.getMouse()
            
            # cars move sequentially
            for car in self.cars:
                if not(car.arrived()):
                    if self.alg == STP:           
                        car.greedy_act() 
                    if self.alg == ASTAR:
                        car.astar_act()

            print(self.test_world.get_map())
new_exp = Experiment(alg = ASTAR, data_file = "data/temp")
new_exp.run()

raw_input("Please Press Enter to Exit")
        
