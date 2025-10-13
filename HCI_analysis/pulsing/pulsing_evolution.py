import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm

import ALPACA.data.finalize as finalize
import numpy as np   

script_dir = os.path.dirname(__file__)
LOAD_WITH_ALPACA = False
if LOAD_WITH_ALPACA:
    CAPTORIUS_BIN_SIZE = 5

    captorius_label_t = 'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*t'
    captorius_label_V = 'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*V'
    pulse_amplitude_label = 'Batman*acq_0*HCI_PulseAmplitude'


    # pulsing runs are 434336:434339
    # HCI_PulseAmplitude scan with values 20, 50, 100, 150
    # additionally we have run 434330 with 190

    data = finalize.generate(first_run=434330, #426447,426472
                            last_run=434339,#426467,426473 
                            elog_results_filename=f'pulsing_scan',
                            known_bad_runs=[r for r in range(434331,434336)],
                            verbosing=True,
                            variables_of_interest=[
                                'Run_Number*Run_Number*__value',  # run number
                                captorius_label_t,
                                captorius_label_V,
                                pulse_amplitude_label
                                ],
                                directories_to_flush=[],
                                speed_mode=False) #'bronze', 'gold','dataset', 'elog'

    # change naming convension from loading to accessing
    captorius_label_t = captorius_label_t.replace('*','_')
    captorius_label_V = captorius_label_V.replace('*','_')
    pulse_amplitude_label = pulse_amplitude_label.replace('*','_')

    # condition data
    captorius = pd.DataFrame(columns=['Run Number','t [s]', 'signal [a.u.]','err_t','err_signal'])
    run_params = pd.DataFrame(columns=['Run Number','pulse_amplitude'])


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

        try:
            run_params = pd.concat([run_params,pd.DataFrame({'Run Number':run_number,'pulse_amplitude':data[pulse_amplitude_label][i]},index=[run_number])])
        except TypeError:
            print('WARNING: missing pulse amplitude infomation')  
    captorius.to_csv(os.path.join(script_dir,'captorius.csv'),index=False)
    run_params.to_csv(os.path.join(script_dir,'run_parameters.csv'),index=False)
    run_params = run_params.set_index('Run Number')
    data = None
else:
    run_params = pd.read_csv(os.path.join(script_dir,'run_parameters.csv'),index_col='Run Number')
    
    captorius = pd.read_csv(os.path.join(script_dir,'captorius.csv'))


runs = run_params.sort_values('pulse_amplitude',ascending=True).index.to_list()

print()
print(runs)
print("Run parameters")
print(run_params)
print(">>>>>>>>>>>>>>>>>>>>")
print("captorius data")
print(captorius)
print(">>>>>>>>>>>>>>>>>>>>")

electrodes = pd.read_csv('electrodes.csv',index_col='label')
print(electrodes)

pulses_sim = pd.read_csv('pulsing.csv',usecols=['position','full']+[f'{pulse:.0f} V' for pulse in run_params['pulse_amplitude']],index_col='position')
print(pulses_sim)

print("data loaded")

################ PLOTS #########################
# plt.rcParams["figure.figsize"] = (3*6.4, 2*4.8)
# fig = plt.figure("Pulsing")
# ax = fig.subplots(len(runs),3, gridspec_kw = {'wspace':0.2, 'hspace':0.0},sharex='col')


# colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink']
# for i, run_number in enumerate(runs):
#     pulse_amplitude = f"{run_params.loc[run_number]['pulse_amplitude']:.0f} V"
#     if i == 0:
#         ax[i,0].set_title('Pulse')
#         ax[i,1].set_title('Pulse (zoom)')
#         ax[i,2].set_title('TOF signal')
#     for j in [0,1]:
#         # ax[i,j].fill_between(pulses_sim.index,pulses_sim['full'],color='black',alpha=0.5)
#         ax[i,j].fill_between(pulses_sim.index,pulses_sim[pulse_amplitude],pulses_sim['full'],color=colors[i],alpha=0.5)
#         ax[i,j].plot(pulses_sim.index,pulses_sim[pulse_amplitude],label=pulse_amplitude,color=colors[i])
#         ax[i,j].plot(pulses_sim.index,pulses_sim['full'],color='black')
#         ax[i,j].set_xlabel('electrode')
#         ax[i,j].set_ylabel('potential [V]')
#         ax[i,j].set_xticks(electrodes['center'])
#         ax[i,j].set_xticks(electrodes['end'],minor=True)
#         ax[i,j].set_xticklabels(electrodes.index.to_list())
#         # ax[i,j].grid(axis='x',which='minor',linestyle = "dashed",linewidth = 0.5,alpha=0.5)
#         # ax[i,j].grid(axis='y',which='major',linewidth = 0.5,alpha=0.5)
#         ax[i,j].grid(False)
#         ax[i,j].tick_params(which = "minor", bottom = False, left = False)
#         ax[i,j].set_xlim(-453.5,-268.5)
#         if j == 0:
#             ax[i,j].set_ylim(0,215)
#         else:
#             ax[i,j].set_ylim(155,205)
#         ax[i,j].legend()

#     ax[i,2].plot(captorius[captorius['Run Number']==run_number]['t [s]'],captorius[captorius['Run Number']==run_number]['signal [a.u.]'],label=run_number,color=colors[i])
#     ax[i,2].set_xlabel('time [s]')
#     ax[i,2].set_ylabel('signal [a.u.]')
#     ax[i,2].set_xlim(1.55e-5,1.9e-5)
#     ax[i,2].set_ylim(-0.45,0.15)
#     ax[i,2].grid()
#     ax[i,2].legend()
#     run_params = pd.read_csv('run_parameters.csv',index_col='Run Number')

plt.rcParams["figure.figsize"] = (2*6.4, 2*4.8)
fig = plt.figure("Pulsing")
ax = fig.subplots(len(runs),2, gridspec_kw = {'wspace':0.2, 'hspace':0.0},width_ratios=[1.5,1],sharex='col')

colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink']
for i, run_number in enumerate(runs):
    pulse_amplitude = f"{run_params.loc[run_number]['pulse_amplitude']:.0f} V"
    if i == 0:
        ax[i,0].set_title('Pulse')
        ax[i,1].set_title('TOF signal')
    # ax[i,j].fill_between(pulses_sim.index,pulses_sim['full'],color='black',alpha=0.5)
    ax[i,0].fill_between(pulses_sim.index,pulses_sim[pulse_amplitude],pulses_sim['full'],color=colors[i],alpha=1.0)
    ax[i,0].plot(pulses_sim.index,pulses_sim[pulse_amplitude],label=pulse_amplitude,color=colors[i])
    ax[i,0].plot(pulses_sim.index,pulses_sim['full'],color='black')
    ax[i,0].set_xlabel('electrode')
    ax[i,0].set_ylabel('potential [V]')
    ax[i,0].set_xticks(electrodes['center'])
    ax[i,0].set_xticks(electrodes['end'],minor=True)
    ax[i,0].set_xticklabels(electrodes.index.to_list())
    # ax[i,j].grid(axis='x',which='minor',linestyle = "dashed",linewidth = 0.5,alpha=0.5)
    # ax[i,j].grid(axis='y',which='major',linewidth = 0.5,alpha=0.5)
    ax[i,0].grid(True,which='minor')
    ax[i,0].tick_params(which = "minor", bottom = False, left = False)
    ax[i,0].set_xlim(electrodes.loc['P7']['start'],electrodes.loc['T1']['end'])
    ax[i,0].set_ylim(0,215)
    ax[i,0].legend(loc='lower left')
    # inset
    axins = ax[i,0].inset_axes([0.6, 0.4, 0.38, 0.58],xlim=(electrodes.loc['P10']['start'],electrodes.loc['P13']['end']), ylim=(140,190))  
    axins.set_xticks(electrodes.loc['P10':'P13']['center'])
    axins.set_xticks(electrodes.loc['P10':'P13']['end'],minor=True)
    axins.set_xticklabels(electrodes.loc['P10':'P13'].index.to_list())
    axins.fill_between(pulses_sim.index,pulses_sim[pulse_amplitude],pulses_sim['full'],color=colors[i],alpha=1.0)
    axins.plot(pulses_sim.index,pulses_sim[pulse_amplitude],label=pulse_amplitude,color=colors[i])
    axins.plot(pulses_sim.index,pulses_sim['full'],color='black')
    axins.grid(True,which='minor')
    ax[i,0].indicate_inset_zoom(axins, edgecolor="black")

    ax[i,1].plot(captorius[captorius['Run Number']==run_number]['t [s]'],captorius[captorius['Run Number']==run_number]['signal [a.u.]'],label=run_number,color=colors[i])
    ax[i,1].set_xlabel('time [s]')
    ax[i,1].set_ylabel('signal [a.u.]')
    ax[i,1].set_xlim(1.55e-5,1.9e-5)
    ax[i,1].set_ylim(-0.35,0.05)
    ax[i,1].grid()
    ax[i,1].legend()
    run_params = pd.read_csv('run_parameters.csv',index_col='Run Number')
fig.tight_layout()
fig.savefig(os.path.join(os.path.dirname(__file__),"pulsing.eps"),dpi=600)
plt.show()

        