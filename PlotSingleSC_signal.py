import os

import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm

import ALPACA.data.finalize as finalize
import numpy as np   

# retrieve the data
run_number = 492042 # 432882 # 434318
sc_index = 12
data = finalize.generate(first_run=run_number, #426447,426472
                        last_run=run_number,#426467,426473 
                        elog_results_filename=f'SC{sc_index}s_{run_number}',
                        known_bad_runs=[],
                        verbosing=True,
                        variables_of_interest=[
                            'Run_Number*Run_Number*__value',  # run number
                            f'SC{sc_index}_coinc*event_clock',
                            f'SC{sc_index}_coinc*event',
                            'SyncCheck_labels',
                            'Sync_check',
                            'labeled_SyncChecks*labels',
                            'labeled_SyncChecks*timestamps_daq_clock_s',
                            'ELENA_Ext_Trigger*acq_0*Timestamp*clock',
                            'ELENA_Ext_Trigger*acq_1*Timestamp*clock',
                            'ELENA_Inj_Trigger*acq_0*Timestamp*clock',
                            'ELENA_Final_Trigger*acq_0*Timestamp*clock',
                            ],
                            directories_to_flush=['bronze','gold','datasets', 'elog'], # 'bronze','gold','datasets', 'elog'
                            speed_mode=True)

# condition data
runs = data['Run_Number_Run_Number___value']
print(runs)

print(data['Sync_check'])
print(data['SyncCheck_labels'])
print(data['labeled_SyncChecks_timestamps_daq_clock_s'])
print(data['labeled_SyncChecks_labels'])
sync_checks = pl.DataFrame({'Run Number':runs,
                            't [s]':data['labeled_SyncChecks_timestamps_daq_clock_s'],
                            'label':data['labeled_SyncChecks_labels']
                            }).explode('t [s]','label').sort('t [s]')
print()
print("sync check")
print(sync_checks)
print(">>>>>>>>>>>>>>>>>>>>")


sc = pl.DataFrame({
    'Run Number':runs,
    't [s]':np.array(data[f'SC{sc_index}_coinc_event_clock']),
    'count':np.array(data[f'SC{sc_index}_coinc_event'])
    }).explode('t [s]','count').sort('t [s]')
print()
print("Scintillator data")
print(sc)
print(">>>>>>>>>>>>>>>>>>>>")

# try:
#     ELENA_triggers = pl.DataFrame({
#         'Run Number':runs,
#         'ELENA_Ext_0':np.array(data['ELENA_Ext_Trigger_acq_0_Timestamp*clock']),
#         'ELENA_Ext_1':np.array(data['ELENA_Ext_Trigger_acq_1_Timestamp*clock']),
#         'ELENA_Inj':np.array(data['ELENA_Inj_Trigger_acq_0_Timestamp*clock']),
#         'ELENA_Final':np.array(data['ELENA_Final_Trigger_acq_0_Timestamp*clock'])
#     })
#     print()
#     print(ELENA_triggers)
# except KeyError|ValueError as e:
#     print(e)

data = None
print("data loaded")

################ PLOTS #########################
sns.set_palette('tab10')
plt.rcParams["figure.figsize"] = (2*6.4, 2*4.8)
# plt.tight_layout()

fig = plt.figure("sc")
t_start = sc.select(pl.col('t [s]').first()).item()
t_end = sc.select(pl.col('t [s]').last()).item()

# move all timestamps
sc = sc.with_columns(pl.col('t [s]') - t_start)
sync_checks = sync_checks.with_columns(pl.col('t [s]') - t_start)

bins = int(t_end-t_start)*100
if bins > 100000:
    bins = 1000

print(f'from {t_start} to {t_end} -> {t_end-t_start} in {bins} bins')
print(f"first sync check = {sync_checks.select(pl.col('t [s]').first()).item()}")

print(sc)
print(sync_checks)
sc_plot = sns.histplot(data=sc,x='t [s]',hue='Run Number',bins=bins,palette='tab10')#,ax=ax[0]) #,weights='count'
sc_plot.grid(axis='x',which='major',linestyle = "dashed",linewidth = 0.5,alpha=0.8)
sc_plot.grid(axis='x',which='minor',linestyle = "dashed",linewidth = 0.5,alpha=0.5)
sc_plot.set_ylim(0,40e3)
# print(f"{bins} -> {hot_storage[i]} ")
# sc_plot.set_xlim(-30,bins + 20 if hot_storage[i] is None else hot_storage[i] + 100)

y_min,y_max = sc_plot.get_ylim()
for index,(clock,label) in enumerate(zip(pl.Series(sync_checks['t [s]']).to_list(),pl.Series(sync_checks['label']).to_list())):
    last_position = 0
    # try:
    label = label
    sync_check_position = clock
    sc_plot.axvline(sync_check_position,lw=1,color='red',alpha=0.5)
    text_position = y_min+(0.2)*(y_max-y_min)
    sc_plot.text(sync_check_position,text_position,label,rotation=90,size='smaller') # rotation=45, # (1+index)*dy
# except:
#         continue
# sc_plot.xaxis.set_major_locator(MultipleLocator(10))
# sc_plot.xaxis.set_minor_locator(MultipleLocator(5))
sc_plot.set_title(f"SC{sc_index}")
# sc_plot.set_xlim()

file = os.path.join(os.path.split(__file__)[0],'images','SC-signal',f'sc{sc_index}_{run_number}.png')
# fig.savefig(file)
plt.show()

        