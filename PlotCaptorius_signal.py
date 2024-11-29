import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm

import ALPACA.data.finalize as finalize
import numpy as np   

# retrieve the data
run_number = 434310 # 432854

CAPTORIUS_BIN_SIZE = 5

captorius_label_t = 'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*t'
captorius_label_V = 'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*V'

data = finalize.generate(first_run=run_number, #426447,426472
                        last_run=run_number,#426467,426473 
                        elog_results_filename=f'captorius_{run_number}',
                        known_bad_runs=[],
                        verbosing=True,
                        variables_of_interest=[
                            'Run_Number*Run_Number*__value',  # run number
                            captorius_label_t,
                            captorius_label_V,
                            ],
                            directories_to_flush=['bronze', 'gold','dataset','elog'],
                            speed_mode=False) #'bronze', 'gold','dataset', 'elog'

# change naming convension from loading to accessing
captorius_label_t = captorius_label_t.replace('*','_')
captorius_label_V = captorius_label_V.replace('*','_')

# condition data
captorius = pd.DataFrame(columns=['Run Number','t [s]', 'signal [a.u.]','err_t','err_signal'])


runs = data['Run_Number_Run_Number___value']
print(runs)

# sync checks
for i,run_number in enumerate(runs):
    print("++++++++++++++++++++++++++++")
    print(run_number)
    print()
    # captorius data
    try:
        index = range(len(data[captorius_label_V][i]))
        captorius_t = data[captorius_label_t][i]
        captorius_V = data[captorius_label_V][i]

        binned_t = []
        binned_V = []
        binned_err_t = []
        binned_err_V = []

        print(captorius_t)
        for bin in range(len(captorius_t)//CAPTORIUS_BIN_SIZE):
            first_bin = bin*CAPTORIUS_BIN_SIZE
            last_bin = first_bin + CAPTORIUS_BIN_SIZE
            binned_t.append(np.median(captorius_t[first_bin:last_bin]))
            binned_err_t.append((captorius_t[last_bin] - captorius_t[first_bin])/2)
            binned_V.append(np.mean(captorius_V[first_bin:last_bin]))
            binned_err_V.append(np.std(captorius_V[first_bin:last_bin]))
        captorius = pd.concat([captorius,pd.DataFrame({'Run Number':[run_number]*len(binned_t),'t [s]':binned_t,'signal [a.u.]':binned_V,'err_t':binned_err_t,'err_signal':binned_err_V})],ignore_index=True)
    except TypeError:
        print('WARNING: missing captorius data')
    

print()
print("captorius data")
print(captorius)
print(">>>>>>>>>>>>>>>>>>>>")



data = None
print("data loaded")


################ PLOTS #########################
sns.set_palette('tab10')
plt.rcParams["figure.figsize"] = (2*6.4, 2*4.8)
# plt.tight_layout()

for i, run_number in enumerate(runs):
    fig = plt.figure(f'{run_number}')
    # plt.errorbar(captorius[captorius['Run Number']==run_number]['t [s]'],captorius[captorius['Run Number']==run_number]['signal [a.u.]'],xerr=captorius[captorius['Run Number']==run_number]['err_t'],yerr=captorius[captorius['Run Number']==run_number]['err_signal'],fmt='o')
    # sns.lineplot(captorius[captorius['Run Number']==run_number],x='t [s]',y='signal [a.u.]',label=f'{run_number}',errorbar='sd') # hue=run_number

    # Draw plot with error bars and extra formatting to match seaborn style
    x = captorius[captorius['Run Number']==run_number]['t [s]']
    y = captorius[captorius['Run Number']==run_number]['signal [a.u.]']
    xerr = captorius[captorius['Run Number']==run_number]['err_t']
    yerr = captorius[captorius['Run Number']==run_number]['err_signal']
    ax = fig.subplots()
    ax.errorbar(x, y, yerr, color='tab:blue', ecolor='tab:blue')
    ax.set_xlabel('timepoint')
    ax.set_ylabel('signal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.show()

        