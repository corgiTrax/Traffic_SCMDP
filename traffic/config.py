import numpy
import random
import copy as cp
import sys

CVXOPT = 0
CVXPY = 1
SOLVER = CVXPY

ROW = 0
COL = 1

#planning horizon
NUM_EPISODE = 20

NUM_CAR = 4000
# car 
STAY = 0; UP = 1; DOWN = 2; LEFT = 3; RIGHT = 4; 
ACTIONS = [STAY, UP, DOWN, LEFT, RIGHT]
# car types
SMALL = 0
BIG = 1
CAP_SMALL = 1
CAP_BIG = 1
CAR_TYPE = [SMALL, BIG]
CAP_CAR = [CAP_SMALL, CAP_BIG]

# Enums for the map
INTERSECT = 0
ROAD = 1
OFFROAD = 2
# number of road blocks between 2 intersections
NUM_BLK_BTW = 1 
# The default car density
DEF_TRAFFIC = 0
# Map
TOTAL_CAP = (NUM_CAR / len(CAR_TYPE) * CAP_SMALL + NUM_CAR / len(CAR_TYPE) * CAP_BIG)
CAP_MAX = int(1 * (NUM_CAR / len(CAR_TYPE) * CAP_SMALL + NUM_CAR / len(CAR_TYPE) * CAP_BIG))
CAP_HZ_ROAD = [1000, 800,  1000]
CAP_VT_ROAD = [1000, 1200, 1000]
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
REWARD = 10
COST = -10
GAMMA = 0.99
# calculate number of cars at each corner
INIT_DENSITY_CORNER = 1.0 / (len(START) * len(CAR_TYPE))

# CONGEST_FACTOR = 1 if we want congest_prob to be 1.0 when current traffic is 200% capacity
# = 0.5 if we want congest_prob to be 1.0 when 300%
# = 2.0 if we want 150%
# increase this value to make penalty more harsh for violating upper bound 
CONGEST_FACTOR = 1.0 
# for planning only
TRANS_SUC_RATE = 1.0

# for visualization
CELL_SIZE = 100
#MOUSE = True

STP = 0 # Shortest path
ASTAR = 1
SCMDPPHI = 2
SCMDPBF = 3
ALGS = [STP, ASTAR, SCMDPPHI, SCMDPBF]
ALG = SCMDPBF
#heuristic to improve efficiency
SCMDP_STP = False
