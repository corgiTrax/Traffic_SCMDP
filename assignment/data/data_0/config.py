NUM_AGENT = 100
NUM_STATE = 11
HOME = 0
REWARD = [0,1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1]

# capacity density upper bound
#CAP_DENSITY = [1.0, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
#CAP_DENSITY = [1.0, 0.02, 0.1, 0.5, 0.01, 0.04, 0.3, 0.2, 0.01, 0.1, 0.2]
CAP_DENSITY = [1.0, 0.2, 0.1, 0.1, 0.2, 0.05, 0.05, 0.2, 0.1, 0.0, 0.0]
INIT_DENSITY = [1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

NUM_EXP = 100
NUM_EPISODE = 31
# allocation algorithms
CENTRALIZED = 0; RANDOM = 1; SAFE = 2; GREEDY = 3
UNC = 4; SCPHI = 5; SCPRO = 6; SCBF = 7; SCUBF = 8
ALGS = [CENTRALIZED, RANDOM, SAFE, GREEDY, UNC, SCPHI, SCPRO, SCBF, SCUBF]
ALGS_NAME = ["CENTRALIZED", "RANDOM", "SAFE", "GREEDY", "UNC", "SCPHI", "SCPRO", "SCBF", "SCUBF"]

# does not use add/drop now
DROP_RATIO = 0.0 # each episode each agent in nonhome patch has probability to be dropped
ADD_RATIO = 1.0 # each episode add some agents to home, but not exceeding initial total number

TRANS_SUC_RATE = 0.75 #0.9 # state transition success rate

CVXOPT = 0
CVXPY = 1
SOLVER = CVXPY

# dir name for where policy files are saved
POLICY_PATH = "policy/"
