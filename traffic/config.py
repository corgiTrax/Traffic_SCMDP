import numpy
import random
import copy as cp
import sys
import math

ROW = 0; COL = 1

# repeat experiments for each algorithm
NUM_EXP = 20
#planning horizon
NUM_EPISODE = 300

# car
#NUM_CARS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000]
NUM_CARS = [20000, 24000, 28000, 32000, 36000, 40000]
NUM_CAR = 1000 #NUM_CARS[2]
STAY = 0; UP = 1; DOWN = 2; LEFT = 3; RIGHT = 4; 
ACTIONS = [STAY, UP, DOWN, LEFT, RIGHT]

# car types
SMALL = 0; BIG = 1
CAP_SMALL = 1; CAP_BIG = 3
CAR_TYPE = [SMALL, BIG]; CAP_CAR = [CAP_SMALL, CAP_BIG]

# Enums for the map
INTERSECT = 0; ROAD = 1; OFFROAD = 2
# number of road blocks between 2 intersections
NUM_BLK_BTW = 1
# The default car density (cars we don't have control, not in use)
DEF_TRAFFIC = 0
# Map
TOTAL_CAP = (NUM_CAR / len(CAR_TYPE) * CAP_SMALL + NUM_CAR / len(CAR_TYPE) * CAP_BIG)
# remove constraint at start and destination states
REMOVE_CONSTRAINT = True
CAP_MAX = TOTAL_CAP 
BETA = 1 # policy was 0.85 for scubf
#CAP_HZ_ROAD = [1000 * BETA, 800 * BETA, 1100 * BETA, 1000 * BETA]
#CAP_VT_ROAD = [1000 * BETA, 1200 * BETA, 900 * BETA, 1000 * BETA]
CAP_HZ_ROAD = [500, 500, 500]
CAP_VT_ROAD = [500, 500, 500]
WORLD_ROWS = (len(CAP_HZ_ROAD) - 1) * (NUM_BLK_BTW + 1) + 1
WORLD_COLS = (len(CAP_VT_ROAD) - 1) * (NUM_BLK_BTW + 1) + 1

# start and destination positions
CAR_COLOR = ["red", "orange", "yellow", "green"]
#START = [[0, 0],[0, WORLD_COLS - 1],[WORLD_ROWS - 1, 0],[WORLD_ROWS - 1, WORLD_COLS - 1]]
#DESTINATION = [[WORLD_ROWS - 1, WORLD_COLS - 1], [WORLD_ROWS - 1, 0], [0, WORLD_COLS - 1],[0, 0]]
#START= [[0,0], [0, WORLD_COLS - 1]]
#DESTINATION = [[WORLD_ROWS - 1, WORLD_COLS - 1], [WORLD_ROWS - 1, 0]]
START= [[0,0]]
DESTINATION = [[WORLD_ROWS - 1, WORLD_COLS - 1]]

# get reward at destination
REWARD = 1
COST = 0
GAMMA = 0.99
# calculate number of cars at each corner
INIT_DENSITY_CORNER = 1.0 / (len(START) * len(CAR_TYPE))

# for planning only
TRANS_SUC_RATE = 1.0

# for visualization
CELL_SIZE = 100

STP = 0 # Shortest path
ASTAR = 1
SCPHI = 2; SCPRO = 3; SCBF = 4; SCUBF = 5; UNC = 6
ALGS = [STP, ASTAR, SCPHI, SCPRO, SCBF, SCUBF]
ALGS_NAME = ["STP", "ASTAR", "SCPHI", "SCPRO", "SCBF", "SCUBF0_85"]
#ALGS = [SCPHI, SCPRO, SCBF, SCUBF]
#ALGS_NAME = ["SCPHI", "SCPRO", "SCBF", "SCUBF"]
ALG = STP
#heuristic to improve efficiency
SCMDP_STP = True


