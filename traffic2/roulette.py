'''A roulette action selector'''

import random
import copy as cp
import numpy

class Roulette:
    ''' input: a probability vector'''
    def __init__(self, prob):
        self.prob = cp.deepcopy(prob)

    def select(self):
        '''return an index that proportional to its probability'''
        num_actions = len(self.prob)
    
        #calc cumulative probability
        cum_prob = numpy.zeros(num_actions + 1)
        cum_prob[0] = 0
        for i in range(num_actions):
            cum_prob[i+1] = cum_prob[i] + self.prob[i]
        
        #random seed
        seed = random.random()
        #print("seed: ", seed)
        #print ["%0.7f" % i for i in cum_prob]
        for i in range(num_actions):
            if (seed >= cum_prob[i] and seed < cum_prob[i+1]):
                action = i
        
        if seed >= cum_prob[-1]: action = num_actions - 1
            
        # print(action)
        return action



