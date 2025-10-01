import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm

import ALPACA.data.finalize as finalize
import numpy as np   

# retrieve the data
first_run = 432877 # 432854
last_run = 432882
sc_index = 12
data = finalize.generate(first_run=first_run, #426447,426472
                        last_run=last_run,#426467,426473 
                        elog_results_filename=f'SC{sc_index}s_{first_run}-{last_run}',
                        known_bad_runs=[],
                        verbosing=True,
                        variables_of_interest=[
                            'Run_Number*Run_Number*__value',  # run number
                            'Sync_check',
                            f'SC{sc_index}_coinc*event_clock',
                            f'SC{sc_index}_coinc*event',
                            'metadata*SyncCheck_labels',
                            ],
                            directories_to_flush=['bronze','gold', 'elog'], # ['bronze','gold','datasets', 'elog']
                            speed_mode=True) #'bronze', 'gold', 'elog'

# condition data
sc = pd.DataFrame(columns=['Run Number','t [s]', 'count'])
sync_checks = pd.DataFrame(columns=['Run Number','clock','label'])

runs = data['Run_Number_Run_Number___value']
print(runs)


# print(data)
# sync checks
for i,run_number in enumerate(runs):
    print(f"++++++++++++++{run_number}++++++++++++++")
    for j, (key,sync_check) in enumerate(data['Sync_check'][i].items()):
        clock = (sync_check['Timestamp']['clock'])*1e-7
        try:
            label = data['metadata_SyncCheck_labels'][i][j]
        except:
            if j==0:
                label = 'pbar_arrival_start'
            elif j==1:
                label = 'pbar_arrival_end'
            elif j==2:
                label = 'hot_dump_start'
            elif j==3:
                label = 'hot_dump_end'
            else:
                label = f'acq_{j}'
        sync_checks = pd.concat([sync_checks,
                                 pd.DataFrame({'Run Number':run_number,
                                               'label':label,
                                               'clock':clock},
                                               index=[0])],
                                               ignore_index=True)
    # scintillator
    try:
        index = range(len(data[f'SC{sc_index}_coinc_event_clock'][0]))
    except TypeError:
        index = [0]
    sc = pd.concat([sc, pd.DataFrame({'Run Number':run_number,'t [s]':np.array(data[f'SC{sc_index}_coinc_event_clock'][0]),'count':np.array(data[f'SC{sc_index}_coinc_event'][0])},index=index)],
            ignore_index=True)

data = None
print("data loaded")

print()
print("sync check")
print(sync_checks)
print(">>>>>>>>>>>>>>>>>>>>")
print()
print("scintillators")
print(sc)


################ PLOTS #########################
plt.rcParams["figure.figsize"] = (2*6.4, 2*4.8)
# plt.tight_layout()

run_to_angle_map = {run_number:angle for (run_number,angle) in zip([432877,432878,432879,432880,432881,432882],[227,217,207,197,187,214])}
counts = []
angle = []

fig_sc = plt.figure("sc")
ax_sc = fig_sc.subplots(len(runs),1,gridspec_kw = {'wspace':0.2, 'hspace':0.01},sharex=True) # plt.figure("sc")
for i,run_number in enumerate(runs):
    first_sync_check = sync_checks.loc[sync_checks["Run Number"]==run_number,"clock"].to_list()[0]
    try:
        pbar_arrival_time = float(sync_checks[(sync_checks["Run Number"]==run_number) & (sync_checks["label"]=='pbar_arrival_start')]["clock"])
        print(pbar_arrival_time)
    except:
        pbar_arrival_time = 0
    sc12 = sc.loc[(sc['Run Number']==run_number),['t [s]']]
    t_start = sc12['t [s]'].to_list()[0]
    t_end = sc12['t [s]'].to_list()[-1]

    print(f'from {t_start} to {t_end} -> {t_end-t_start}')
    print(f'first sync check = {first_sync_check}')
    bins = int(t_end-t_start)*10
    if bins > 100000:
        bins = 1000
    print(f"Data with {len(sc12['t [s]'])} is arranged in {bins}")
    # plotting raw scintillator signal
    ax_sc[i].hist(sc12['t [s]'] - pbar_arrival_time,bins=bins,fill=True,label=run_number,color='tab:blue')
    ax_sc[i].set_xlabel('t [s]')
    ax_sc[i].set_ylabel('counts')
    ax_sc[i].set_ylim(0,40e3)

    for index,sync_check in sync_checks[sync_checks['Run Number']==run_number].iterrows():
        try:
            label = sync_check['label']
            ax_sc[i].axvline(sync_check['clock']-pbar_arrival_time,lw=1,color='red',alpha=0.5)
            ax_sc[i].text(sync_check['clock']-pbar_arrival_time,0+0.2*(40e3-0),label,rotation=90,size='smaller') # rotation=45, # (1+index)*dy
        except:
            continue
    ax_sc[i].xaxis.set_major_locator(MultipleLocator(10))
    ax_sc[i].xaxis.set_minor_locator(MultipleLocator(5))

file = os.path.join(os.path.split(__file__)[0],'plots',f'SC56_{run_number}.png')
fig_sc.savefig(file)
plt.show()

        