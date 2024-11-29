import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm

import ALPACA.data.finalize as finalize
import numpy as np   

# retrieve the data
# first working sync_check 426370
# testing mu vs s vs SyncCheck line: 433652-433655
run_number = 433030 # 433030 # 432854 # 426370
data = finalize.generate(first_run=433652, 
                        last_run=433655, # 433655
                        elog_results_filename=f'syncchecks_{run_number}',
                        known_bad_runs=[],
                        verbosing=True,
                        variables_of_interest=[
                            'Run_Number*Run_Number*__value',  # run number
                            'Sync_check',
                            'metadata',
                            ],
                            directories_to_flush=[ 'bronze', 'gold','elog'],
                            speed_mode=True) #'bronze', 'gold', 'datasets', 'elog'
                            # directories_to_flush=['bronze','gold','datasets', 'elog'],
                            # speed_mode=False) #'bronze', 'gold', 'datasets', 'elog'

# condition data
sync_checks = pd.DataFrame(columns=['Run Number','t [s]'])
sync_checks_labels = pd.DataFrame(columns=['Run Number','Label','t [s]','t [mu]'])

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
                                    pd.DataFrame({'Run Number':run_number,'Label':f'{key}','t [s]':sync_check['Timestamp']['clock']*1e-7}, #  'Name':sync_check_label_map[j],
                                                index=[0])],
                                    ignore_index=True)

    times_mu = []
    times_s = []
    labels = []
    for j,(label,values) in enumerate(data['metadata'][i].items()):
        if not label.startswith('SyncCheck_'):
            continue
        label = label.removeprefix('SyncCheck_')
        for value in values:
            if label.startswith('mu_'):
                times_mu.append(value)
                label = label.removeprefix('mu_')
                labels.append(label)
            elif label.startswith('s_'):
                times_s.append(value)
                label.removeprefix('s_')



    print(labels)
    print(times_mu)
    print(times_s)
    sync_checks_labels = pd.concat([sync_checks_labels,pd.DataFrame({'Run Number':[run_number for _ in range(len(labels))],'Label':labels,'t [s]':times_s,'t [mu]':times_mu})],ignore_index=True)
sync_checks_labels = sync_checks_labels.sort_values('t [mu]').reset_index(drop=True)

print()
print("sync check")
print(sync_checks)
print(">>>>>>>>>>>>>>>>>>>>")


print()
print("sync check with labels")
print(sync_checks_labels)
print(">>>>>>>>>>>>>>>>>>>>")
data = None
print("data loaded")

################ PLOTS #########################
plt.rcParams["figure.figsize"] = (2*6.4, 2*4.8)
# plt.tight_layout()

fig = plt.figure("syncchecks")
ax = fig.subplots(2,1,gridspec_kw = {'wspace':0.2, 'hspace':0.01})
for run_number in runs:
    ax[0].plot(sync_checks_labels[sync_checks_labels["Run Number"]==run_number]['t [mu]'],sync_checks_labels[sync_checks_labels["Run Number"]==run_number]['t [s]'], '.-', label=run_number)
    # ax[0].plot(sync_checks_labels[sync_checks["Run Number"]==run_number]['t [mu]'],sync_checks_labels[sync_checks_labels["Run Number"]==run_number]['t [s]'], '.--', label=run_number)
ax[0].set_xlabel('t [mu]')
ax[0].set_ylabel('t [s]')
ax[0].legend()


delta = []
real_deltas = [100000,250000,10] # ns
for run_number in runs:
    print(run_number)

    first_mu  = sync_checks_labels[(sync_checks_labels["Run Number"]==run_number) & (sync_checks_labels["Label"]=="First")]['t [mu]'].iloc[0]
    second_mu = sync_checks_labels[(sync_checks_labels["Run Number"]==run_number) & (sync_checks_labels["Label"]=="Second")]['t [mu]'].iloc[0]
    third_mu = sync_checks_labels[(sync_checks_labels["Run Number"]==run_number) & (sync_checks_labels["Label"]=="Third")]['t [mu]'].iloc[0]
    fourth_mu = sync_checks_labels[(sync_checks_labels["Run Number"]==run_number) & (sync_checks_labels["Label"]=="Fourth")]['t [mu]'].iloc[0]

    first_s  = sync_checks_labels[(sync_checks_labels["Run Number"]==run_number) & (sync_checks_labels["Label"]=="First")]['t [s]'].iloc[0]
    second_s = sync_checks_labels[(sync_checks_labels["Run Number"]==run_number) & (sync_checks_labels["Label"]=="Second")]['t [s]'].iloc[0]
    third_s = sync_checks_labels[(sync_checks_labels["Run Number"]==run_number) & (sync_checks_labels["Label"]=="Third")]['t [s]'].iloc[0]
    fourth_s = sync_checks_labels[(sync_checks_labels["Run Number"]==run_number) & (sync_checks_labels["Label"]=="Fourth")]['t [s]'].iloc[0]

    delta1_mu = second_mu - first_mu
    delta1_s = second_s - first_s
    print(f'{delta1_mu} vs {delta1_s*1e9} vs {real_deltas[0]}')

    delta2_mu = third_mu - second_mu
    delta2_s = third_s - second_s
    print(f'{delta2_mu} vs {delta2_s*1e9} vs {real_deltas[1]}')

    delta3_mu = fourth_mu - third_mu
    delta3_s = fourth_s - third_s
    print(f'{delta3_mu} vs {delta3_s*1e9} vs {real_deltas[2]}')




plt.show()