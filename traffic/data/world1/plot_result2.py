import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import scipy.stats as ss
import math
NUM_EXP = 50
NUM_EPISODE = 50
NUM_CARS = [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500]#, 10000]
CAPS = []
for car in NUM_CARS:
    CAPS.append(car * 2)

def process_file(filename):
    data_file = open(filename, 'r')
    arrived = 0
    for i, line in enumerate(data_file):
        if i == NUM_EPISODE - 2: # remember last step there's no decision
            data = re.split(',|\n', line)
            arrived = float(data[1]) #/ NUM_EPISODE
    data_file.close()
    return arrived

def process_data(name):
    arr = []
    
    for car in NUM_CARS:
        arr_ = []
        for i in range(NUM_EXP):
            filename = str(car) +'/' + name + str(i) + ".data"
            arr_.append(process_file(filename))
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
ax1.errorbar(CAPS, stp_arr_mean, yerr=ss.t.ppf(0.95, stp_dof)*stp_arr_std, color = 'black', fmt = '^', ls = 'dotted', label="STP")
ax1.errorbar(CAPS, astar_arr_mean, yerr=ss.t.ppf(0.95, astar_dof)*astar_arr_std, color = 'blue', fmt = '>', ls = 'dotted', label = "TB-A*")
#ax1.errorbar(CAPS, un_arr_mean, yerr=ss.t.ppf(0.95, un_dof)*un_arr_std, color = 'pink', fmt = '+', ls = 'dotted', label = "UNC")
ax1.errorbar(CAPS, scphi_arr_mean, yerr=ss.t.ppf(0.95, scphi_dof)*scphi_arr_std, color = 'm', fmt = 'x', ls = 'dotted', label = "SCWC")
ax1.errorbar(CAPS, scpro_arr_mean, yerr=ss.t.ppf(0.95, scpro_dof)*scpro_arr_std, color = 'orange', fmt = 'o', ls = 'dotted', label = "SCPRO")
ax1.errorbar(CAPS, scbf_arr_mean, yerr=ss.t.ppf(0.95, scbf_dof)*scbf_arr_std, color = 'g', fmt = 'D', ls = 'dotted', label = "SCBF")
ax1.errorbar(CAPS, scubf_arr_mean, yerr=ss.t.ppf(0.95, scubf_dof)*scubf_arr_std, color = 'r', fmt = 's', ls = 'dotted', label = "SCF")

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc = 'upper left')
plt.ylim((-1000, None))
#plt.xlim((-5, 3800))
plt.xlabel("Total Vehicle Capacity at Starting Position", fontsize = 24)
plt.ylabel("Capacity Arrived within 50 steps", fontsize = 24)
font = {'size': 16}
plt.rc('font', **font)
plt.show()
# ax1.set_yscale('symlog', linthreshy = 10)#nonposy='clip')

# plot final arrival % of total cars fixed number of steps over # of cars 


