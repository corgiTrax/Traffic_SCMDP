import numpy
import random
import copy as cp
import sys
import math

ROW = 0; COL = 1

#planning horizon
NUM_EPISODE = 30

# car 
NUM_CAR = 200
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
REMOVE_CONSTRAINT = False
CAP_MAX = TOTAL_CAP 
CAP_HZ_ROAD = [100, 100]
CAP_VT_ROAD = [100, 100]
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
UNC = 2; SCPHI = 3; SCPRO = 4; SCBF = 5; SCUBF = 6
ALGS = [STP, ASTAR, UNC, SCPHI, SCPRO, SCBF, SCUBF]
ALGS_NAME = ["STP", "ASTAR", "UNC", "SCPHI", "SCPRO", "SCBF", "SCUBF"]
ALG = SCUBF
#heuristic to improve efficiency
SCMDP_STP = True


