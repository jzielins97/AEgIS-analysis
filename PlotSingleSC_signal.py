import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm

import ALPACA.data.finalize as finalize
import numpy as np   

# retrieve the data
run_number = 432882 # 434318
sc_index = 12
data = finalize.generate(first_run=run_number, #426447,426472
                        last_run=run_number,#426467,426473 
                        elog_results_filename=f'SC{sc_index}s_{run_number}',
                        known_bad_runs=[],
                        verbosing=True,
                        variables_of_interest=[
                            'Run_Number*Run_Number*__value',  # run number
                            'Sync_check',
                            f'SC{sc_index}_coinc*event_clock',
                            f'SC{sc_index}_coinc*event',
                            'metadata*SyncCheck_labels',
                            ],
                            directories_to_flush=[ 'bronze', 'gold','elog'],
                            speed_mode=True) #'bronze', 'gold', 'elog'
                            # directories_to_flush=['bronze','gold','datasets', 'elog'],
                            # speed_mode=False) #'bronze', 'gold', 'datasets', 'elog'

# condition data
sc = pd.DataFrame(columns=['Run Number','t [s]', 'count'])
sync_checks = pd.DataFrame(columns=['Run Number','clock','label'])

runs = data['Run_Number_Run_Number___value']
print(runs)


print(data['metadata_SyncCheck_labels'])
# sync checks
for i,run_number in enumerate(runs):
    print("++++++++++++++++++++++++++++")
    print()
    print(run_number)
    for j, (key,sync_check) in enumerate(data['Sync_check'][i].items()):

        clock = (sync_check['Timestamp']['clock'])*1e-7
        try:
            label = data['metadata_SyncCheck_labels'][i][j]
        except:
            label = f'acq_{j}'
        sync_checks = pd.concat([sync_checks,
                                 pd.DataFrame({'Run Number':run_number,
                                               'label':label,
                                               'clock':clock},
                                               index=[0])],
                                               ignore_index=True)

print()
print("sync check")
print(sync_checks)
print(">>>>>>>>>>>>>>>>>>>>")

# scintillator
try:
    index = range(len(data[f'SC{sc_index}_coinc_event_clock'][0]))
except TypeError:
    index = [0]
sc = pd.concat([sc, pd.DataFrame({'Run Number':run_number,'t [s]':np.array(data[f'SC{sc_index}_coinc_event_clock'][0]),'count':np.array(data[f'SC{sc_index}_coinc_event'][0])},index=index)],
        ignore_index=True)

data = None
print("data loaded")
print(sc)


################ PLOTS #########################
sns.set_palette('tab10')
plt.rcParams["figure.figsize"] = (2*6.4, 2*4.8)
# plt.tight_layout()

fig = plt.figure("sc")
# fig,ax = plt.subplots(1,2,gridspec_kw = {'wspace':0.2, 'hspace':0.01}) # plt.figure("sc")

first_sync_check = sync_checks[sync_checks["Run Number"]==run_number]["clock"].iloc[0]
try:
    pbar_arrival_time = float(sync_checks[(sync_checks["Run Number"]==run_number) & (sync_checks["label"]=='pbar_arrival_start')]["clock"])
    print(pbar_arrival_time)
except:
    pbar_arrival_time = first_sync_check
sc_data = sc[(sc['Run Number']==run_number)] # & (sc['t [s]'] < 115292150460)
t_start = sc_data['t [s]'].iloc[0]
t_end = sc_data['t [s]'].iloc[-1]

print(f'from {t_start} to {t_end} -> {t_end-t_start}')
print(f'first sync check = {first_sync_check}')
sc_data['t [s]'] -= pbar_arrival_time
bins = int(t_end-t_start)*10
if bins > 100000:
    bins = 1000
print(len(sc[sc['Run Number']==run_number]['t [s]']))
print(bins)
sc_plot = sns.histplot(data=sc_data,x='t [s]',hue='Run Number',bins=bins,palette='tab10')#,ax=ax[0]) #,weights='count'
sc_plot.grid(axis='x',which='major',linestyle = "dashed",linewidth = 0.5,alpha=0.8)
sc_plot.grid(axis='x',which='minor',linestyle = "dashed",linewidth = 0.5,alpha=0.5)
sc_plot.set_ylim(0,40e3)
# print(f"{bins} -> {hot_storage[i]} ")
# sc_plot.set_xlim(-30,bins + 20 if hot_storage[i] is None else hot_storage[i] + 100)

y_min,y_max = sc_plot.get_ylim()
dy = 1.0/len(sync_checks[sync_checks['Run Number']==run_number])
for index,sync_check in sync_checks[sync_checks['Run Number']==run_number].iterrows():
    last_position = 0
    try:
        label = sync_check['label']
        sync_check_position = sync_check['clock']-pbar_arrival_time
        sc_plot.axvline(sync_check_position,lw=1,color='red',alpha=0.5)
        text_position = y_min+(0.2)*(y_max-y_min)
        sc_plot.text(sync_check_position,text_position,label,rotation=90,size='smaller') # rotation=45, # (1+index)*dy
        # if label == 'acq_0':
        #     sc_plot.text(sync_check_position,text_position,'pbar_arrival',rotation=90,size='smaller') # rotation=45, # (1+index)*dy
        # elif label == 'acq_2':
        #     sc_plot.text(sync_check_position,text_position,'pbar_dump_start',rotation=90,size='smaller') # rotation=45, # (1+index)*
        # elif label == 'acq_3':
        #     sc_plot.text(sync_check_position,text_position,'pbar_dump_end',rotation=90,size='smaller') # rotation=45, # (1+index)*dy
    except:
        continue
sc_plot.xaxis.set_major_locator(MultipleLocator(10))
sc_plot.xaxis.set_minor_locator(MultipleLocator(5))
sc_plot.set_title(f"SC{sc_index}")

file = os.path.join(os.path.split(__file__)[0],'images','SC-signal',f'sc{sc_index}_{run_number}.png')
# fig.savefig(file)
plt.show()

        