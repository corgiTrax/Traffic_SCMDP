# Author: Christian Careaga (christian.careaga7@gmail.com)
# A* Pathfinding in Python (2.7)
# Please give credit if used

import numpy
import random
from heapq import *

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

#def heuristic(a, b):
#    return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2
def heuristic(a, b):
    return abs(b[0] - a[0])  + abs(b[1] - a[1]) 

neighbors = [(0,1),(0,-1),(1,0),(-1,0)] #,(1,1),(1,-1),(-1,1),(-1,-1)]
random.shuffle(neighbors)

def astar_path(array, start, goal):

    close_set = set()
    came_from = {}
    gscore = {start:0}
    fscore = {start:heuristic(start, goal)}
    oheap = []

    heappush(oheap, (fscore[start], start))
    
    while oheap:

        current = heappop(oheap)[1]

        if current == goal:
            data = []
            while current in came_from:
                data.append(current)
                current = came_from[current]
            return data

        close_set.add(current)
        
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j 
            # time based astar algorithm: add transition cost
            transit_cost = 0
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    transit_cost = array[neighbor[0]][neighbor[1]]
                    if transit_cost == 1: transit_cost = 0 # do not consider walls
            tentative_g_score = gscore[current] + heuristic(current, neighbor) + transit_cost # add another cost here
            if 0 <= neighbor[0] < array.shape[0]:
                if 0 <= neighbor[1] < array.shape[1]:                
                    if array[neighbor[0]][neighbor[1]] == 1:
                        continue
                else:
                    # array bound y walls
                    continue
            else:
                # array bound x walls
                continue
                
            if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                continue

            if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [k[1]for k in oheap]:
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heappush(oheap, (fscore[neighbor], neighbor))
              
    return []

'''Here is an example of using my algo with a numpy array,
   astar(array, start, destination)
   astar function returns a list of points (shortest path)'''

if __name__ == "__main__":
#    nmap = numpy.array([
#        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#        [1,1,1,1,1,1,1,1,1,1,1,1,0,1],
#        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#        [1,0,1,1,1,1,1,1,1,1,1,1,1,1],
#        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#        [1,1,1,1,1,1,1,1,1,1,1,1,0,1],
#        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#        [1,0,1,1,1,1,1,1,1,1,1,1,1,1],
#        [0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#        [1,1,1,1,1,1,1,1,1,1,1,1,0,1],
#       [0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

    nmap = numpy.array([
    [0,     0,      0,      0,      0],
    [0,     1,      0,      1,      0],
    [0,     0,      0,      0,      0],
    [0,     1,      0,      1,      0],
    [0,     0,      0,      0,      0]
    ])

    print_matrix(nmap)
    print(astar_path(nmap, (2,2), (0,0)))
