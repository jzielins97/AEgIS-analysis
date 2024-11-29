import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm

import ALPACA.data.finalize as finalize
import numpy as np   

# retrieve the data
run_number = 432854
data = finalize.generate(first_run=run_number, #426447,426472
                        last_run=run_number,#426467,426473 
                        elog_results_filename=f'SC56_{run_number}',
                        known_bad_runs=[],
                        verbosing=True,
                        variables_of_interest=[
                            'Run_Number*Run_Number*__value',  # run number
                            # 'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*t',
                            # 'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*V',
                            # 'captorius1*acq_0*Channel_3_TOF_ions*Y_[V]*t',
                            # 'captorius1*acq_0*Channel_2_TOF_ions*Y_[V]*V',
                            # 'captorius1*acq_0*Channel_1_TOF_electrons*Y_[V]*t',
                            # 'captorius1*acq_0*Channel_1_TOF_electrons*Y_[V]*V',
                            'Sync_check',
                            'SC56_coinc*event_clock',
                            'SC56_coinc*event',
                            # 'SC56_coinc*t_0'
                            '1TCMOS*acq_1*C_flatten_data',
                            'Metadata',
                            # '1TCMOS*acq_1*height',
                            # '1TCMOS*acq_1*width',
                            # 'Batman*acq_0*Catch_HotStorageTime'
                            ],
                            directories_to_flush=['bronze','gold','datasets', 'elog'],
                            speed_mode=True) #'bronze', 'gold', 'datasets', 'elog'
                            # directories_to_flush=['bronze','gold','datasets', 'elog'],
                            # speed_mode=False) #'bronze', 'gold', 'datasets', 'elog'

# Captorius 1 Manager_Config*0*Captorius 1 Manager_Config*Configuration_Name_*__value -> this should give if it is ions or electrons

# condition data
sync_checks = pd.DataFrame(columns=['Run Number','Label','Time [s]'])
sc = pd.DataFrame(columns=['Run Number','t [s]', 'count'])
captorius = pd.DataFrame(columns=['Run Number','channel','t [s]', 'signal [a.u.]'])
# cmos = pd.DataFrame(columns=['Run Number','x [px]','y [px]','signal [a.u.]'])


print(run_number)


sync_check_label_map = ['PBar arrival start','PBar arrival end','HV off start','HV off stop','ions dump','reset ptrap 1','reset ptrap 2']
# sync checks
N = len(data['Sync_check'][0])
# sync_checks = pd.concat([sync_checks,pd.DataFrame({'Run Number':run_number,'Label':'acq_0','Time [s]':data['Sync_check'][i]['acq_0']['Timestamp']['clock']*1e-9},index=[0])],ignore_index=True)
for j in range(N):
    sync_checks = pd.concat([sync_checks,
                                pd.DataFrame({'Run Number':run_number,'Label':f'acq_{j}','Time [s]':data['Sync_check'][0][f'acq_{j}']['Timestamp']['clock']*1e-7}, #  'Name':sync_check_label_map[j],
                                            index=[0])],
                                ignore_index=True)
    
# scintillator
try:
    index = range(len(data['SC56_coinc_event_clock'][0]))
except TypeError:
    index = [0]
sc = pd.concat([sc, pd.DataFrame({'Run Number':run_number,'t [s]':np.array(data['SC56_coinc_event_clock'][0]),'count':np.array(data['SC56_coinc_event'][0])},index=index)],
        ignore_index=True)

# captorius data
try:
    index = range(len(data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_V'][0]))
except TypeError:
    index = [0]
captorius = pd.concat([captorius,pd.DataFrame({'Run Number':run_number,'channel':0,'t [s]':data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_t'][0],'signal [a.u.]':data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_V'][0]},index=index)],
                    ignore_index=True)

print()
print("sync check")
print(sync_checks)
print(">>>>>>>>>>>>>>>>>>>>")

data = None
print("data loaded")


################ PLOTS #########################
sns.set_palette('tab10')
plt.rcParams["figure.figsize"] = (2*6.4, 2*4.8)
# plt.tight_layout()

fig = plt.figure("sc")
# fig,ax = plt.subplots(1,2,gridspec_kw = {'wspace':0.2, 'hspace':0.01}) # plt.figure("sc")

first_sync_check = sync_checks[sync_checks["Run Number"]==run_number]["Time [s]"].iloc[0]
t_start = sc[sc['Run Number']==run_number]['t [s]'].iloc[0]
t_end = sc[sc['Run Number']==run_number]['t [s]'].iloc[-1]

print(f'from {t_start} to {t_end} -> {t_end-t_start}')
# print(f'first sync check = {first_sync_check}')
sc.loc[sc['Run Number']==run_number,'t [s]'] = sc[sc['Run Number']==run_number]['t [s]'] - first_sync_check
bins = int(t_end-t_start)
if bins > 100000:
    bins = 1000
# print(len(sc[sc['Run Number']==run_number]['t [s]']))
# print(bins)
sc_plot = sns.histplot(data=sc[sc['Run Number']==run_number],x='t [s]',hue='Run Number',bins=bins,palette='tab10')#,ax=ax[0]) #,weights='count'
sc_plot.grid(axis='x',which='major',linestyle = "dashed",linewidth = 0.5,alpha=0.8)
sc_plot.grid(axis='x',which='minor',linestyle = "dashed",linewidth = 0.5,alpha=0.5)
sc_plot.set_ylim(0,10000)
# print(f"{bins} -> {hot_storage[i]} ")
# sc_plot.set_xlim(-30,bins + 20 if hot_storage[i] is None else hot_storage[i] + 100)
for time in sync_checks[sync_checks['Run Number']==run_number]['Time [s]']:
    try:
        sc_plot.axvline(time-first_sync_check,lw=1,color='red',alpha=0.5)
    except:
        continue
sc_plot.xaxis.set_major_locator(MultipleLocator(10))
sc_plot.xaxis.set_minor_locator(MultipleLocator(5))

# if len(captorius[captorius['Run Number']==run_number]) > 1:
#     # print(len(captorius[captorius['Run Number']==run_number]['t [s]']))
#     # t_start = captorius[captorius['Run Number']==run_number]['t [s]'].iloc[0]*1e6 # us
#     # t_end = captorius[captorius['Run Number']==run_number]['t [s]'].iloc[-1]*1e6 # us
#     tof_bins = 1000 #int(t_end-t_start)
#     # tof_plot = sns.lineplot(data=captorius[captorius['Run Number']==run_number],x='t [s]',y='signal [a.u.]',hue='Run Number',palette='tab10',ax=sc_ax[i,1]) # ,log_scale=[False,True] # ,weights='count'
#     tof_plot = sns.histplot(data=captorius[captorius['Run Number']==run_number],x='t [s]',weights='signal [a.u.]',bins=tof_bins,hue='Run Number',palette='tab10',ax=ax[1], element="poly", fill=False) # ,log_scale=[False,True] # ,weights='count'
#     tof_plot.grid(axis='x',which='major',linestyle = "dashed",linewidth = 0.5,alpha=0.8)
#     # tof_plot.grid(axis='x',which='minor',linestyle = "dashed",linewidth = 0.5,alpha=0.5)
#     tof_plot.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
#     # tof_plot.set_ylim(-10,10)
#     # tof_plot.set_xlim(-0.00002,0.00012)
#     tof_plot.set_xlim(0,0.00002)
#     tof_plot.xaxis.set_major_locator(MultipleLocator(0.00002))
#     tof_plot.xaxis.set_minor_locator(MultipleLocator(0.000005))



file = os.path.join(os.path.split(__file__)[0],'plots',f'SC56_{run_number}.png')
fig.savefig(file)
plt.show()

        