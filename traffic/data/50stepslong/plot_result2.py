import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import scipy.stats as ss
import math
NUM_EXP = 10
NUM_EPISODE = 50
NUM_CARS = [1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000]#, 17000, 19000, 21000, 23000, 25000, 27000, 29000]
CAPS = []
for car in NUM_CARS:
    CAPS.append(car * 2)

def process_file(filename, cap):
    data_file = open(filename, 'r')
    arrived = 0
    for i, line in enumerate(data_file):
        if i == 25: #NUM_EPISODE - 2: # remember last step there's no decision
            data = re.split(',|\n', line)
            arrived = float(data[1]) / 1
    data_file.close()
    return arrived

def process_data(name):
    arr = []
    
    for cap in CAPS:
        arr_ = []
        for i in range(NUM_EXP):
            filename = str(cap) +'/' + name + str(i) + ".data"
            arr_.append(process_file(filename, cap))
        arr.append(cp.deepcopy(arr_))
    
    arr = np.array(arr)
    arr_mean = np.mean(arr, axis = 1)
    arr_std = np.std(arr, axis = 1)
    dof = np.empty(len(CAPS))
    dof.fill(NUM_EXP)
  
    return dof, arr_mean, arr_std

stp_dof, stp_arr_mean, stp_arr_std = process_data("STP")
astar_dof, astar_arr_mean, astar_arr_std = process_data("ASTAR")
un_dof, un_arr_mean, un_arr_std = process_data("UNC")
scphi_dof, scphi_arr_mean, scphi_arr_std = process_data("SCPHI")
scpro_dof, scpro_arr_mean, scpro_arr_std = process_data("SCPRO")
scbf_dof, scbf_arr_mean, scbf_arr_std = process_data("SCBF")
scubf_dof, scubf_arr_mean, scubf_arr_std = process_data("SCUBF")

# plot arrival over episodes
fig, (ax1) = plt.subplots(1,1)
ax1.errorbar(CAPS, stp_arr_mean, yerr=ss.t.ppf(0.95, stp_dof)*stp_arr_std, color = 'b', fmt = '^', ls = 'dotted', label="STP")
ax1.errorbar(CAPS, astar_arr_mean, yerr=ss.t.ppf(0.95, astar_dof)*astar_arr_std, color = 'c', fmt = '>', ls = 'dotted', label = "ASTAR")
#ax1.errorbar(CAPS, un_arr_mean, yerr=ss.t.ppf(0.95, un_dof)*un_arr_std, color = 'pink', fmt = '+', ls = 'dotted', label = "UNC")
ax1.errorbar(CAPS, scphi_arr_mean, yerr=ss.t.ppf(0.95, scphi_dof)*scphi_arr_std, color = 'm', fmt = 'x', ls = 'dotted', label = "SCPHI")
ax1.errorbar(CAPS, scpro_arr_mean, yerr=ss.t.ppf(0.95, scpro_dof)*scpro_arr_std, color = 'orange', fmt = 'o', ls = 'dotted', label = "SCPRO")
ax1.errorbar(CAPS, scbf_arr_mean, yerr=ss.t.ppf(0.95, scbf_dof)*scbf_arr_std, color = 'g', fmt = 'D', ls = 'dotted', label = "SCBF")
ax1.errorbar(CAPS, scubf_arr_mean, yerr=ss.t.ppf(0.95, scubf_dof)*scubf_arr_std, color = 'r', fmt = '*', ls = 'dotted', label = "SCUBF")

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc = 'upper left')
#plt.ylim((-0.1, 1.2))
plt.show()
# ax1.set_yscale('symlog', linthreshy = 10)#nonposy='clip')

# plot final arrival % of total cars fixed number of steps over # of cars 


