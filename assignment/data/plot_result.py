import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import scipy.stats as ss
import math
dirname = sys.argv[1]
NUM_EXP = int(sys.argv[2])
# plot only every INTERVAL element, not in use
# INTERVAL = int(sys.argv[3])

def skip_array(array):
    n = len(array)
    new_array = np.zeros(n/INTERVAL + 1)
    for i in range(n):
        if i % INTERVAL == 0:
            new_array[i / INTERVAL] = array[i]
    return new_array

def process_file(filename):
    data_file = open(filename, 'r')
    episodes = []
    rewards = []
    violations = []
    line_count = 0
    for line in data_file:
        if line_count != 0:
            data = re.split(',|\n', line)
            episodes.append(float(data[0])) 
            rewards.append(float(data[1]))
            violations.append(float(data[2]))
        line_count += 1
    data_file.close()
    return episodes, rewards, violations

def process_data(name):
    eps = []
    rs = []
    vs = []

    for i in range(NUM_EXP):
        filename = dirname + name + str(i) + ".data"
        eps_, rs_, vs_ = process_file(filename)
        eps.append(cp.deepcopy(eps_)) 
        rs.append(cp.deepcopy(rs_))
        vs.append(cp.deepcopy(vs_))
    
    eps = np.array(eps[0]) 
    rs = np.array(rs)
    vs = np.array(vs)
    rs_mean = np.mean(rs, axis = 0)
    rs_std = np.std(rs, axis = 0)
    vs_mean = np.mean(vs, axis = 0)
    vs_std = np.std(vs, axis = 0)
    dof = np.empty(len(eps))
    dof.fill(NUM_EXP)
    
#    eps = skip_array(eps)
#    rs_mean = skip_array(rs_mean)
#    rs_std = skip_array(rs_std)
#    vs_mean = skip_array(vs_mean)
#    vs_std = skip_array(vs_std)
#    dof = skip_array(dof)

    return dof, eps, rs_mean, rs_std, vs_mean, vs_std

cent_dof, cent_eps, cent_rs_mean, cent_rs_std, cent_vs_mean, cent_vs_std = process_data("CENTRALIZED")
rand_dof, rand_eps, rand_rs_mean, rand_rs_std, rand_vs_mean, rand_vs_std = process_data("RANDOM")
safe_dof, safe_eps, safe_rs_mean, safe_rs_std, safe_vs_mean, safe_vs_std = process_data("SAFE")
grdy_dof, grdy_eps, grdy_rs_mean, grdy_rs_std, grdy_vs_mean, grdy_vs_std = process_data("GREEDY")
un_dof, un_eps, un_rs_mean, un_rs_std, un_vs_mean, un_vs_std = process_data("UNC")
scphi_dof, scphi_eps, scphi_rs_mean, scphi_rs_std, scphi_vs_mean, scphi_vs_std = process_data("SCPHI")
scpro_dof, scpro_eps, scpro_rs_mean, scpro_rs_std, scpro_vs_mean, scpro_vs_std = process_data("SCPRO")
scbf_dof, scbf_eps, scbf_rs_mean, scbf_rs_std, scbf_vs_mean, scbf_vs_std = process_data("SCBF")
scubf_dof, scubf_eps, scubf_rs_mean, scubf_rs_std, scubf_vs_mean, scubf_vs_std = process_data("SCUBF")

# plot rewards
fig, (ax1) = plt.subplots(1,1)
ax1.errorbar(cent_eps, cent_rs_mean, yerr=ss.t.ppf(0.95, cent_dof)*cent_rs_std, color = '0.6', fmt = '^', ls = 'dotted', label="Centralized")
ax1.errorbar(rand_eps, rand_rs_mean, yerr=ss.t.ppf(0.95, rand_dof)*rand_rs_std, color = 'b', fmt = 'v', ls = 'dotted', label = "Random")
ax1.errorbar(safe_eps, safe_rs_mean, yerr=ss.t.ppf(0.95, safe_dof)*safe_rs_std, color = 'k', fmt = '<', ls = 'dotted', label = "Safe")
ax1.errorbar(grdy_eps, grdy_rs_mean, yerr=ss.t.ppf(0.95, grdy_dof)*grdy_rs_std, color = 'c', fmt = '>', ls = 'dotted', label = "Greedy")
#ax1.errorbar(un_eps, un_rs_mean, yerr=ss.t.ppf(0.95, un_dof)*un_rs_std, color = 'pink', fmt = '+', ls = 'dotted', label = "UNC")
ax1.errorbar(scphi_eps, scphi_rs_mean, yerr=ss.t.ppf(0.95, scphi_dof)*scphi_rs_std, color = 'm', fmt = 'x', ls = 'dotted', label = "SCPHI")
ax1.errorbar(scpro_eps, scpro_rs_mean, yerr=ss.t.ppf(0.95, scpro_dof)*scpro_rs_std, color = 'orange', fmt = 'o', ls = 'dotted', label = "SCPRO")
ax1.errorbar(scbf_eps, scbf_rs_mean, yerr=ss.t.ppf(0.95, scbf_dof)*scbf_rs_std, color = 'g', fmt = 'D', ls = 'dotted', label = "SCBF")
ax1.errorbar(scubf_eps, scubf_rs_mean, yerr=ss.t.ppf(0.95, scubf_dof)*scubf_rs_std, color = 'r', fmt = 's', ls = 'dotted', label = "SCF")
handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc = 'upper left')
plt.ylim((-1,2100))
plt.xlabel("Episode")
plt.ylabel("Total Reward Collected")
plt.show()

# plot violations
fig, (ax1) = plt.subplots(1,1)
ax1.errorbar(cent_eps, cent_vs_mean, yerr=ss.t.ppf(0.95, cent_dof)*cent_vs_std, color = '0.6', fmt = '^', ls = 'dotted', label="Centralized")
ax1.errorbar(rand_eps, rand_vs_mean, yerr=ss.t.ppf(0.95, rand_dof)*rand_vs_std, color = 'b', fmt = 'v', ls = 'dotted', label = "Random")
ax1.errorbar(safe_eps, safe_vs_mean, yerr=ss.t.ppf(0.95, safe_dof)*safe_vs_std, color = 'k', fmt = '<', ls = 'dotted', label = "Safe")
ax1.errorbar(grdy_eps, grdy_vs_mean, yerr=ss.t.ppf(0.95, grdy_dof)*grdy_vs_std, color = 'c', fmt = '>', ls = 'dotted', label = "Greedy")
#ax1.errorbar(un_eps, un_vs_mean, yerr=ss.t.ppf(0.95, un_dof)*un_vs_std, color = 'pink', fmt = '+', ls = 'dotted', label = "UNC")
ax1.errorbar(scphi_eps, scphi_vs_mean, yerr=ss.t.ppf(0.95, scphi_dof)*scphi_vs_std, color = 'm', fmt = 'x', ls = 'dotted', label = "SCPHI")
ax1.errorbar(scpro_eps, scpro_vs_mean, yerr=ss.t.ppf(0.95, scpro_dof)*scpro_vs_std, color = 'orange', fmt = 'o', ls = 'dotted', label = "SCPRO")
ax1.errorbar(scbf_eps, scbf_vs_mean, yerr=ss.t.ppf(0.95, scbf_dof)*scbf_vs_std, color = 'g', fmt = 'D', ls = 'dotted', label = "SCBF")
ax1.errorbar(scubf_eps, scubf_vs_mean, yerr=ss.t.ppf(0.95, scubf_dof)*scubf_vs_std, color = 'r', fmt = 's', ls = 'dotted', label = "SCF")

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles, labels, loc = 'upper left')
# ax1.set_yscale('symlog', linthreshy = 10)#nonposy='clip')
#plt.xlim((-1,None))
plt.ylim((-50,None))
plt.xlabel("Episode")
plt.ylabel("Total Violation Count")
plt.show()


