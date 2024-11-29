import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm

import ALPACA.data.finalize as finalize
import numpy as np

# seaborn setup
# 426396
# 426396
# 426440

# [426396,426397,426424,426440]
# bad_runs = [426404,426405,426407,426408,426409,42410,426411,426421,426425]

# omited_runs = [omited_run for omited_run in range(426398,426485)]
omited_runs = [omited_run for omited_run in range(427037,427038)]    

# retrieve the data
data = finalize.generate(first_run=427148, #427136
                        last_run=427153,#427153
                        elog_results_filename='HCI_boiloff',
                        known_bad_runs= [426587,426589,426600] + omited_runs,
                        verbosing=True,
                        variables_of_interest=[
                            'Run_Number*Run_Number*__value',  # run number
                            'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*t',
                            'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*V',
                            'captorius1*acq_0*Channel_1_TOF_electrons*Y_[V]*t',
                            'captorius1*acq_0*Channel_1_TOF_electrons*Y_[V]*V',
                            #  'captorius1*acq_0*Channel_3_TOF_ions*Y_[V]*t',
                            #  'captorius1*acq_0*Channel_2_TOF_ions*Y_[V]*V',
                            #  'captorius1*acq_0*Channel_3_TOF_ions*Y_[V]*t',
                            #  'captorius1*acq_0*Channel_3_TOF_ions*Y_[V]*V',
                            'Sync_check',
                            'SC56_coinc*event_clock',
                            'SC56_coinc*event',
                            # 'SC56_coinc*t_0'
                            '1TCMOS*acq_1*background_corrected*background_normalised_img',
                            '1TCMOS*acq_1*height',
                            '1TCMOS*acq_1*width',
                            'Batman*acq_0*Catch_HotStorageTime',
                            'Batman*acq_0*Pbar_CoolingTime'
                            'Batman*acq_0*NestedTrap_SqueezeTime',
                            'Batman*acq_0*NestedTrap_RaiseTime',
                            'Batman*acq_0*NestedTrap_SqueezedTrapType',
                            ],
                            directories_to_flush=['bronze', 'gold', 'datasets', 'elog'],
                            speed_mode=False) #'bronze', 'gold', 'datasets', 'elog'
                            # directories_to_flush=[],
                            # speed_mode=False) #'bronze', 'gold', 'datasets', 'elog'


# condition data
sync_checks = pd.DataFrame(columns=['Run Number','Label','acq','Time [s]'])
sc = pd.DataFrame(columns=['Run Number','t [s]', 'count'])
captorius = pd.DataFrame(columns=['Run Number','channel','t [s]', 'signal [a.u.]'])
cmos = pd.DataFrame(columns=['Run Number','x [px]','y [px]','signal [a.u.]'])



runs = np.array(data['Run_Number_Run_Number___value'])
hot_storage = np.array(data['Batman_acq_0_Catch_HotStorageTime'])
print(runs)


sync_check_label_map = ['PBar arrival start','PBar arrival end','HV off start','HV off stop','ions dump','reset ptrap 1','reset trap']
for i,run_number in enumerate(runs):
    # sync checks
    try:
        N = len(data['Sync_check'][i])
        for j in range(N):
            sync_checks = pd.concat([sync_checks,
                                    pd.DataFrame({'Run Number':run_number,'acq':j,'Time [s]':data['Sync_check'][i][f'acq_{j}']['Timestamp']['clock']*1e-7}, #'Label':sync_check_label_map[j],
                                                index=[0])],
                                    ignore_index=True)
    except TypeError:
        print(f'No sync checks for run {run_number}')
        
    # scintillator
    try:
        index = range(len(data['SC56_coinc_event_clock'][i]))
        sc = pd.concat([sc, pd.DataFrame({'Run Number':run_number,'t [s]':np.array(data['SC56_coinc_event_clock'][i]),'count':np.array(data['SC56_coinc_event'][i])},index=index)],
            ignore_index=True)
    except TypeError:
        print(f"No scintilator data available for {run_number}")
    
    # captorius data
    try:
        index = range(len(data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_V'][i]))
        captorius = pd.concat([captorius,pd.DataFrame({'Run Number':run_number,'channel':0,'t [s]':data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_t'][i],'signal [a.u.]':data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_V'][i]},index=index)],
                        ignore_index=True)
    except TypeError:
        print(f"No captorius (ions) data for run {run_number}")
    
    try:
        index = range(len(data['captorius1_acq_0_Channel_1_TOF_electrons_Y_[V]_V'][i]))
        captorius = pd.concat([captorius,pd.DataFrame({'Run Number':run_number,'channel':0,'t [s]':data['captorius1_acq_0_Channel_1_TOF_electrons_Y_[V]_t'][i],'signal [a.u.]':data['captorius1_acq_0_Channel_1_TOF_electrons_Y_[V]_V'][i]},index=index)],
                        ignore_index=True)
    except TypeError:
        print(f"No captorius (electron) data for run {run_number}")
    
    # # CMOS
    try:
        x_px = [x for _ in range(data['1TCMOS_acq_1_height'][i]) for x in range(data['1TCMOS_acq_1_width'][i])]
        y_px = [y for y in range(data['1TCMOS_acq_1_height'][i]) for _ in range(data['1TCMOS_acq_1_width'][i])]
        cmos = pd.concat([cmos,pd.DataFrame({'Run Number':run_number,'x [px]':x_px,'y [px]':y_px,'signal [a.u.]':[float(val) for val in np.reshape(data['1TCMOS_acq_1_background_corrected_background_normalised_img'][i],len(x_px))]})],
                        ignore_index=True)
    except TypeError:
        print(f"Nb CMOS data for run {run_number}")


    # for val in np.array(cmos['signal [a.u.]']):
    #     print(val)
# exit(0)
print()
print("sync check")
print(sync_checks)
print(">>>>>>>>>>>>>>>>>>>>")

# print()
# print("scintillator")
# print(sc)
# print(">>>>>>>>>>>>>>>>>>>>")

# print()
# print("captorius")
# print(captorius)
# print(">>>>>>>>>>>>>>>>>>>>")

# print()
# print("cmos")
# print(cmos)
# print(">>>>>>>>>>>>>>>>>>>>")

data = None
print("data loaded")


################ PLOTS #########################
sns.set_palette('tab10')
plt.rcParams["figure.figsize"] = (2*6.4, 2*4.8)
# plt.tight_layout()

fig, ax = plt.subplots(len(runs),3,gridspec_kw = {'wspace':0.2, 'hspace':0.1}) # plt.figure("sc")

for i,run_number in enumerate(runs):
    sc_ax= ax[i,0] if len(runs) > 1 else ax[0]
    first_sync_check = sync_checks[sync_checks["Run Number"]==run_number]["Time [s]"].iloc[0]
    t_start = sc[sc['Run Number']==run_number]['t [s]'].iloc[0]
    t_end = sc[sc['Run Number']==run_number]['t [s]'].iloc[-1]
    
    print(f'from {t_start} to {t_end} -> {t_end-t_start}')
    print(f'first sync check = {first_sync_check}')
    sc.loc[sc['Run Number']==run_number,'t [s]'] = sc[sc['Run Number']==run_number]['t [s]'] - first_sync_check
    # bins=100
    bins = int(t_end-t_start)
    if bins > 100000:
        bins = 1000
    print(len(sc[sc['Run Number']==run_number]['t [s]']))
    print(bins)
    sc_plot = sns.histplot(data=sc[sc['Run Number']==run_number],x='t [s]',hue='Run Number',bins=bins,palette='tab10',ax=sc_ax) #,weights='count'
    sc_plot.grid(axis='x',which='major',linestyle = "dashed",linewidth = 0.5,alpha=0.8)
    sc_plot.grid(axis='x',which='minor',linestyle = "dashed",linewidth = 0.1,alpha=0.5)
    sc_plot.set_ylim(0,500) # 2e8)
    # print(f"{bins} -> {hot_storage[i]} ")
    # sc_plot.set_xlim(-30,bins + 20 if hot_storage[i] is None else hot_storage[i] + 100)
    # sc_plot.set_xlim(-30,1900)
    for time in sync_checks[sync_checks['Run Number']==run_number]['Time [s]']:
        try:
            sc_plot.axvline(time-first_sync_check,lw=1,color='red',alpha=0.5)
        except:
            continue
    sc_plot.xaxis.set_major_locator(MultipleLocator(60))
    sc_plot.xaxis.set_minor_locator(MultipleLocator(10))

    if len(captorius[captorius['Run Number']==run_number]) != 1:
        captorius_ax= ax[i,1] if len(runs) > 1 else ax[1]
        # print(len(captorius[captorius['Run Number']==run_number]['t [s]']))
        # t_start = captorius[captorius['Run Number']==run_number]['t [s]'].iloc[0]*1e6 # us
        # t_end = captorius[captorius['Run Number']==run_number]['t [s]'].iloc[-1]*1e6 # us
        tof_bins = 1000 #int(t_end-t_start)
        # tof_plot = sns.lineplot(data=captorius[captorius['Run Number']==run_number],x='t [s]',y='signal [a.u.]',hue='Run Number',palette='tab10',ax=sc_ax[i,1]) # ,log_scale=[False,True] # ,weights='count'
        tof_plot = sns.histplot(data=captorius[captorius['Run Number']==run_number],x='t [s]',weights='signal [a.u.]',bins=tof_bins,hue='Run Number',palette='tab10',ax=captorius_ax, element="poly", fill=False) # ,log_scale=[False,True] # ,weights='count'
        tof_plot.grid(axis='x',which='major',linestyle = "dashed",linewidth = 0.5,alpha=0.8)
        # tof_plot.grid(axis='x',which='minor',linestyle = "dashed",linewidth = 0.5,alpha=0.5)
        tof_plot.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        # tof_plot.set_ylim(-10,10)
        # tof_plot.set_xlim(-0.00002,0.00012)
        tof_plot.set_xlim(-0.0000,10e-6)
        # tof_plot.xaxis.set_major_locator(MultipleLocator(0.00002))
        # tof_plot.xaxis.set_minor_locator(MultipleLocator(0.000005))



    # plot CMOS picture
    cmos_ax = ax[i,2] if len(runs) > 1 else ax[2]
    cmos_plot = sns.histplot(cmos[cmos['Run Number']==run_number], x='x [px]',y='y [px]',
                            discrete=(True,True),
                            # palette="viridis",
                            stat='count',
                            weights='signal [a.u.]',
                            cbar=True,
                            norm=LogNorm(),
                            vmin=None,
                            vmax=None,
                            ax=cmos_ax)


file = os.path.join(os.path.split(__file__)[0],'plots','HCI_2024',f'overview_{runs[0]}-{runs[-1]}.png')
fig.savefig(file)
plt.show()

        