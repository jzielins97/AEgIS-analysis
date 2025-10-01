import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm

import ALPACA.data.finalize as finalize
import numpy as np   

# retrieve the data
runs = [378934,395665,432882,478040]
data = finalize.generate(first_run=runs[0], #433821, 
                        last_run=runs[-1], # 433825,
                        elog_results_filename='syncchecks_v3',
                        known_bad_runs=[n for n in range(runs[0],runs[-1]) if n not in runs],
                        verbosing=True,
                        variables_of_interest=[
                            'Run_Number*Run_Number*__value',  # run number
                            'labeled_SyncChecks*labels',
                            'labeled_SyncChecks*timestamps_kasli_s',
                            'labeled_SyncChecks*timestamps_daq_clock',
                            'labeled_SyncChecks*timestamps_daq_clock_s',
                            ],
                            directories_to_flush=['bronze','gold','datasets','elog'], # 'bronze','gold','datasets','elog'
                            speed_mode=False) #'bronze', 'gold', 'datasets', 'elog'

# print(data)
runs = data['Run_Number_Run_Number___value']
print(runs)
labels = data['labeled_SyncChecks_labels']
timestamps_daq_clock = data['labeled_SyncChecks_timestamps_daq_clock']
timestamps_daq_clock_s = data['labeled_SyncChecks_timestamps_daq_clock_s']
timestamps_kasli_s = data['labeled_SyncChecks_timestamps_kasli_s']

for idx,run_number in enumerate(runs):
    print(f"-------{run_number}---------")
    print(labels[idx],timestamps_daq_clock[idx],timestamps_daq_clock_s[idx],timestamps_kasli_s[idx],sep='\n')

# ################ PLOTS #########################
# plt.rcParams["figure.figsize"] = (2*6.4, 2*4.8)
# colors = ['b','g','r','c','m','y']
# markers = ['x','+',"s",'D']
# fig = plt.figure("syncchecks")
# real_deltas = [100000,250000,10] # ns
# ax = fig.subplots(2,1) # ,gridspec_kw = {'wspace':0.2, 'hspace':0.01}
# ax[0].plot([sum(real_deltas[0:i]) + i*1e5 if i!=0 else 0 for i in range(len(real_deltas)+1)],[0,0,0,0],'x',label="function calls",color='black')
# ax[0].set_xlabel('relative time [ns]')
# ax[0].set_ylabel('different approach')
# for i,run_number in enumerate(runs):
#     ax[0].plot((sync_checks[sync_checks["Run Number"]==run_number]['clock'] - sync_checks[sync_checks["Run Number"]==run_number]['clock'].iloc[0])*100,[1 for _ in range(len(sync_checks[sync_checks["Run Number"]==run_number]['clock']))], 's', color=colors[i], label=f'sync check line {run_number}')
#     ax[0].plot(sync_checks[sync_checks["Run Number"]==run_number]['rtio_counter_mu'] - sync_checks[sync_checks["Run Number"]==run_number]['rtio_counter_mu'].iloc[0],[2 for _ in range(len(sync_checks[sync_checks["Run Number"]==run_number]['rtio_counter_mu']))], 'D', color=colors[i], label=f'mu {run_number}')
#     ax[0].plot((sync_checks[sync_checks["Run Number"]==run_number]['rtio_counter_s']-sync_checks[sync_checks["Run Number"]==run_number]['rtio_counter_s'].iloc[0])*1e9,[3 for _ in range(len(sync_checks[sync_checks["Run Number"]==run_number]['rtio_counter_s']))], 'o', color=colors[i], label=f's {run_number}')
#     if i == 0:
#         ax[0].legend()

# for i,run_number in enumerate(runs):
#     ax[1].plot(sync_checks[sync_checks["Run Number"]==run_number]['clock'],sync_checks[sync_checks["Run Number"]==run_number]['rtio_counter_mu']-sync_checks[sync_checks["Run Number"]==run_number]['clock'], color=colors[i], label=run_number)

# ax[1].set_ylabel('rtio counter (mu) - sync check clock')
# ax[1].set_xlabel('sync check clock')
# ax[1].legend()

# plt.show()