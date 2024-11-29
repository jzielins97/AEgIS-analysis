import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm

import ALPACA.data.finalize as finalize
import numpy as np   

# retrieve the data
data = finalize.generate(first_run=434079, #433821, 
                        last_run=434079, # 433825,
                        elog_results_filename='syncchecks_v2',
                        known_bad_runs=[],
                        verbosing=True,
                        variables_of_interest=[
                            'Run_Number*Run_Number*__value',  # run number
                            'Sync_check',
                            'metadata*SyncCheck_labels',
                            'metadata*SyncCheck_s',
                            'metadata*SyncCheck_mu'
                            ],
                            directories_to_flush=[ 'bronze', 'gold','datasets','elog'],
                            speed_mode=True) #'bronze', 'gold', 'datasets', 'elog'

# condition data
sync_checks = pd.DataFrame(columns=['Run Number','clock','label','rtio_counter_mu','rtio_counter_s'])

runs = data['Run_Number_Run_Number___value']
print(runs)


print(data)
# sync checks
for i,run_number in enumerate(runs):
    print("++++++++++++++++++++++++++++")
    print()
    print(run_number)
    for j, (key,sync_check) in enumerate(data['Sync_check'][i].items()):
        sync_checks = pd.concat([sync_checks,
                                 pd.DataFrame({'Run Number':run_number,
                                               'label':data['metadata_SyncCheck_labels'][i][j],
                                               'clock':(sync_check['Timestamp']['clock']), # clock is 10 Mhz
                                               'rtio_counter_mu':(data['metadata_SyncCheck_mu'][i][j]), # 1 mu is 1 ns 
                                               'rtio_counter_s':(data['metadata_SyncCheck_s'][i][j])}, # this value is in seconds
                                               index=[0])],
                                               ignore_index=True)

print()
print("sync check")
print(sync_checks)
print(">>>>>>>>>>>>>>>>>>>>")

data = None
print("data loaded")
################ PLOTS #########################
plt.rcParams["figure.figsize"] = (2*6.4, 2*4.8)
colors = ['b','g','r','c','m','y']
markers = ['x','+',"s",'D']
fig = plt.figure("syncchecks")
real_deltas = [100000,250000,10] # ns
ax = fig.subplots(2,1) # ,gridspec_kw = {'wspace':0.2, 'hspace':0.01}
ax[0].plot([sum(real_deltas[0:i]) + i*1e5 if i!=0 else 0 for i in range(len(real_deltas)+1)],[0,0,0,0],'x',label="function calls",color='black')
ax[0].set_xlabel('relative time [ns]')
ax[0].set_ylabel('different approach')
for i,run_number in enumerate(runs):
    ax[0].plot((sync_checks[sync_checks["Run Number"]==run_number]['clock'] - sync_checks[sync_checks["Run Number"]==run_number]['clock'].iloc[0])*100,[1 for _ in range(len(sync_checks[sync_checks["Run Number"]==run_number]['clock']))], 's', color=colors[i], label=f'sync check line {run_number}')
    ax[0].plot(sync_checks[sync_checks["Run Number"]==run_number]['rtio_counter_mu'] - sync_checks[sync_checks["Run Number"]==run_number]['rtio_counter_mu'].iloc[0],[2 for _ in range(len(sync_checks[sync_checks["Run Number"]==run_number]['rtio_counter_mu']))], 'D', color=colors[i], label=f'mu {run_number}')
    ax[0].plot((sync_checks[sync_checks["Run Number"]==run_number]['rtio_counter_s']-sync_checks[sync_checks["Run Number"]==run_number]['rtio_counter_s'].iloc[0])*1e9,[3 for _ in range(len(sync_checks[sync_checks["Run Number"]==run_number]['rtio_counter_s']))], 'o', color=colors[i], label=f's {run_number}')
    if i == 0:
        ax[0].legend()

for i,run_number in enumerate(runs):
    ax[1].plot(sync_checks[sync_checks["Run Number"]==run_number]['clock'],sync_checks[sync_checks["Run Number"]==run_number]['rtio_counter_mu']-sync_checks[sync_checks["Run Number"]==run_number]['clock'], color=colors[i], label=run_number)

ax[1].set_ylabel('rtio counter (mu) - sync check clock')
ax[1].set_xlabel('sync check clock')
ax[1].legend()

plt.show()