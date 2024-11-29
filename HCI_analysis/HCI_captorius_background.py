import os

import pandas as pd
import seaborn as sns
import scipy as sc
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm

import ALPACA.data.finalize as finalize
import numpy as np

def fit_func(x, amplitude=1, phi=0, offset=0):
    return offset + amplitude*np.sin(x - phi)
    
# retrieve the data
data = finalize.generate(first_run=426554,
                        last_run=426563, 
                        elog_results_filename='HCI_captorius_backgrounds',
                        known_bad_runs= [],
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
# print("captorius")
# print(captorius)
# print(">>>>>>>>>>>>>>>>>>>>")

# print()
# print("cmos")
# print(cmos)
# print(">>>>>>>>>>>>>>>>>>>>")

data = None
print(captorius.dtypes)
captorius = captorius.astype({'Run Number':pd.Int32Dtype(),'channel':pd.Int8Dtype(),'t [s]':pd.Float64Dtype(),'signal [a.u.]':pd.Float64Dtype()})
print(captorius.dtypes)
captorius.to_parquet('bcg_captorius.parquet')
print("data loaded")

print(captorius)

# fit captorius backgrounds with sin
bcg_fit = pd.DataFrame(columns=['Run Number','amplitude','phi','offset'])
for i,run_number in enumerate(runs):
    popt, pcov = sc.curve_fit(fit_func,captorius[captorius['Run Number']==run_number]['t [s]'],captorius[captorius['Run Number']==run_number]['signal [a.u.]'],p0=[1,0,0])
    print(popt,pcov)
    bcg_fit = pd.concat([bcg_fit,pd.DataFrame({'Run Number':run_number,'amplitude':popt[0],'phi':popt[1],'offset':popt[2]},index=0)],ignore_index=True)

print(bcg_fit)
exit()


################ PLOTS #########################
sns.set_palette('tab10')
plt.rcParams["figure.figsize"] = (2*6.4, 2*4.8)
# plt.tight_layout()

# captorius_bcg = captorius['t [s]','signal [a.u.]'].groupby('t [s]').mean()



fig, ax = plt.subplots(2,gridspec_kw = {'wspace':0.2, 'hspace':0.01}) # plt.figure("sc")

sns.histplot(captorius,x='t [s]',weights='signal [a.u.]',bins=1000,palette='tab10',hue="Run Number",element="poly",fill=False,ax=ax[0])
sns.histplot(captorius.groupby('t [s]').mean(),x='t [s]',bins=1000,element="poly",fill=False,ax=ax[1],kde=True,line_kws={'color':'black'}, color='blue') # ,



# file = os.path.join(os.path.split(__file__)[0],'plots','HCI_2024',f'bcg_{runs[0]}-{runs[1]}.png')
# bcg.savefig(file)

# fig, ax = plt.subplots(len(runs),2,gridspec_kw = {'wspace':0.2, 'hspace':0.01}) # plt.figure("sc")

# for i,run_number in enumerate(runs):

#     if len(captorius[captorius['Run Number']==run_number]) == 1:
#         continue
#     tof_bins = 1000 #int(t_end-t_start)
#     # tof_plot = sns.lineplot(data=captorius[captorius['Run Number']==run_number],x='t [s]',y='signal [a.u.]',hue='Run Number',palette='tab10',ax=sc_ax[i,1]) # ,log_scale=[False,True] # ,weights='count'
#     tof_plot = sns.histplot(data=captorius[captorius['Run Number']==run_number],x='t [s]',weights='signal [a.u.]',bins=tof_bins,hue='Run Number',palette='tab10',ax=ax[i,0], element="poly", fill=False) # ,log_scale=[False,True] # ,weights='count'
#     tof_plot.grid(axis='x',which='major',linestyle = "dashed",linewidth = 0.5,alpha=0.8)
#     # tof_plot.grid(axis='x',which='minor',linestyle = "dashed",linewidth = 0.5,alpha=0.5)
#     tof_plot.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
#     # tof_plot.set_ylim(-10,10)
#     tof_plot.set_xlim(-0.00002,0.00012)
#     ax[i,0].xaxis.set_major_locator(MultipleLocator(0.00002))
#     ax[i,0].xaxis.set_minor_locator(MultipleLocator(0.000005))
#     if i == 0:
#         tof_plot.set_title('signal')

    



# file = os.path.join(os.path.split(__file__)[0],'plots','HCI_2024',f'bcg_runs_{runs[0]}-{runs[1]}.png')
# fig.savefig(file)
plt.show()

        