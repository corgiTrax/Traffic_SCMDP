import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import scipy.stats as ss
import math
dirname = sys.argv[1]
#TOTAL_CAP = float(sys.argv[2])
NUM_EXP = int(sys.argv[2])
CAP = int(sys.argv[3]) * 2 # input is the num of cars
# plot only every INTERVAL element
INTERVAL = int(sys.argv[4])

def skip_array(array):
    n = len(array)
    new_array = []
    for i in range(n):
        if i % INTERVAL == 0:
            new_array.append(array[i])
    return new_array

def process_file(filename):
    data_file = open(filename, 'r')
    episodes = []
    arrived = []
    for line in data_file:
        data = re.split(',|\n', line)
        episodes.append(float(data[0])) 
        arrived.append(float(data[1])/CAP)
    data_file.close()
    return episodes, arrived

def process_data(name):
    eps = []
    arr = []

    for i in range(NUM_EXP):
        filename = dirname + name + str(i) + ".data"
        eps_, arr_ = process_file(filename)
        eps.append(cp.deepcopy(eps_)) 
        arr.append(cp.deepcopy(arr_))
    
    eps = np.array(eps[0]) 
    arr = np.array(arr)
    arr_mean = np.mean(arr, axis = 0)
    arr_std = np.std(arr, axis = 0)
    dof = np.empty(len(eps))
    dof.fill(NUM_EXP)
    
    eps = skip_array(eps)
    arr_mean = skip_array(arr_mean)
    arr_std = skip_array(arr_std)
    dof = skip_array(dof)

    return dof, eps, arr_mean, arr_std

stp_dof, stp_eps, stp_arr_mean, stp_arr_std = process_data("STP")
astar_dof, astar_eps, astar_arr_mean, astar_arr_std = process_data("ASTAR")
un_dof, un_eps, un_arr_mean, un_arr_std = process_data("UNC")
scphi_dof, scphi_eps, scphi_arr_mean, scphi_arr_std = process_data("SCPHI")
scpro_dof, scpro_eps, scpro_arr_mean, scpro_arr_std = process_data("SCPRO")
scbf_dof, scbf_eps, scbf_arr_mean, scbf_arr_std = process_data("SCBF")
scubf_dof, scubf_eps, scubf_arr_mean, scubf_arr_std = process_data("SCUBF")

# plot arrival over episodes
fig, (ax1) = plt.subplots(1,1)
ax1.errorbar(stp_eps, stp_arr_mean, yerr=ss.t.ppf(0.95, stp_dof)*stp_arr_std, color = 'black', fmt = '^', ls = 'dotted', label="STP")
ax1.errorbar(astar_eps, astar_arr_mean, yerr=ss.t.ppf(0.95, astar_dof)*astar_arr_std, color = 'blue', fmt = '>', ls = 'dotted', label = "TB-ASTAR")
#ax1.errorbar(un_eps, un_arr_mean, yerr=ss.t.ppf(0.95, un_dof)*un_arr_std, color = 'pink', fmt = '+', ls = 'dotted', label = "UNC")
ax1.errorbar(scphi_eps, scphi_arr_mean, yerr=ss.t.ppf(0.95, scphi_dof)*scphi_arr_std, color = 'm', fmt = 'x', ls = 'dotted', label = "SCWC")
ax1.errorbar(scpro_eps, scpro_arr_mean, yerr=ss.t.ppf(0.95, scpro_dof)*scpro_arr_std, color = 'orange', fmt = 'o', ls = 'dotted', label = "SCPRO")
ax1.errorbar(scbf_eps, scbf_arr_mean, yerr=ss.t.ppf(0.95, scbf_dof)*scbf_arr_std, color = 'g', fmt = 'D', ls = 'dotted', label = "SCBF")
ax1.errorbar(scubf_eps, scubf_arr_mean, yerr=ss.t.ppf(0.95, scubf_dof)*scubf_arr_std, color = 'r', fmt = 's', ls = 'dotted', label = "SCF")

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc = 4, framealpha = 0.5)
#plt.ylim((None, 1.5))
plt.xlabel("Time Step", fontsize = 24)
plt.ylabel("Ratio of Capacity Arrivied", fontsize = 24)
font = {'size': 16}
plt.rc('font', **font)
plt.show()
# ax1.set_yscale('symlog', linthreshy = 10)#nonposy='clip')

# plot final arrival % of total cars fixed number of steps over # of cars 


