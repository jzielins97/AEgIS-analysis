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

omited_runs = [omited_run for omited_run in range(426398,426485)]
    
# retrieve the data
data = finalize.generate(first_run=426529, #426447,426472
                        last_run=426544,#426467,426473 
                        elog_results_filename='HCI_signal',
                        known_bad_runs= [426474,426475,426476,426477, 426545] + omited_runs,
                        verbosing=True,
                        variables_of_interest=[
                            'Run_Number*Run_Number*__value',  # run number
                            'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*t',
                            'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*V',
                            #  'captorius1*acq_0*Channel_3_TOF_ions*Y_[V]*t',
                            #  'captorius1*acq_0*Channel_2_TOF_ions*Y_[V]*V',
                            #  'captorius1*acq_0*Channel_3_TOF_ions*Y_[V]*t',
                            #  'captorius1*acq_0*Channel_3_TOF_ions*Y_[V]*V',
                            # 'Sync_check',
                            # 'SC56_coinc*event_clock',
                            # 'SC56_coinc*event',
                            # 'SC56_coinc*t_0'
                            '1TCMOS*acq_1*C_flatten_data',
                            '1TCMOS*acq_1*height',
                            '1TCMOS*acq_1*width',
                            # 'Batman*acq_0*Catch_HotStorageTime'
                            ],
                            # directories_to_flush=['bronze', 'gold', 'datasets', 'elog'],
                            # speed_mode=True) #'bronze', 'gold', 'datasets', 'elog'
                            directories_to_flush=[],
                            speed_mode=False) #'bronze', 'gold', 'datasets', 'elog'


# condition data
captorius = pd.DataFrame(columns=['Run Number','channel','t [s]', 'signal [a.u.]'])
cmos = pd.DataFrame(columns=['Run Number','x [px]','y [px]','signal [a.u.]'])

runs = np.array(data['Run_Number_Run_Number___value'])
print(runs)

for i,run_number in enumerate(runs):    
    # captorius data
    try:
        index = range(len(data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_V'][i]))
    except TypeError:
        index = [0]
    captorius = pd.concat([captorius,pd.DataFrame({'Run Number':run_number,'channel':0,'t [s]':data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_t'][i],'signal [a.u.]':data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_V'][i]},index=index)],
                        ignore_index=True)
    
    # CMOS
    x_px = [x for _ in range(data['1TCMOS_acq_1_height'][i]) for x in range(data['1TCMOS_acq_1_width'][i])]
    y_px = [y for y in range(data['1TCMOS_acq_1_height'][i]) for _ in range(data['1TCMOS_acq_1_width'][i])]
    cmos = pd.concat([cmos,pd.DataFrame({'Run Number':run_number,'x [px]':x_px,'y [px]':y_px,'signal [a.u.]':[float(val) for val in data['1TCMOS_acq_1_C_flatten_data'][i]]})],
                     ignore_index=True)


# print()
# print("sync check")
# print(sync_checks)
# print(">>>>>>>>>>>>>>>>>>>>")

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

fig, ax = plt.subplots(1,2,gridspec_kw = {'wspace':0.2, 'hspace':0.01})
tof_bins = 1000 #int(t_end-t_start)
tof_plot = sns.histplot(data=captorius,x='t [s]',weights='signal [a.u.]',bins=tof_bins,ax=ax[0], element="poly", fill=False) # ,log_scale=[False,True] # ,weights='count'
tof_plot.grid(axis='x',which='major',linestyle = "dashed",linewidth = 0.5,alpha=0.8)
# tof_plot.grid(axis='x',which='minor',linestyle = "dashed",linewidth = 0.5,alpha=0.5)
tof_plot.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
# tof_plot.set_ylim(-10,10)
tof_plot.set_xlim(-0.00002,0.00012)
ax[0].xaxis.set_major_locator(MultipleLocator(0.00002))
ax[0].xaxis.set_minor_locator(MultipleLocator(0.000005))

# plot CMOS picture
cmos_plot = sns.histplot(cmos, x='x [px]',y='y [px]',
                        discrete=(True,True),
                        # palette="viridis",
                        stat='count',
                        weights='signal [a.u.]',
                        cbar=True,
                        norm=LogNorm(),
                        vmin=None,
                        vmax=None,
                        ax=ax[1])


file = os.path.join(os.path.split(__file__)[0],'plots','HCI_2024',f'captorius_combined_{runs[0]}-{runs[-1]}.png')
fig.savefig(file)
plt.show()

        