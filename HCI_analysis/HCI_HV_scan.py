import os

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.colors import LogNorm

import ALPACA.data.finalize as finalize
import numpy as np   


LOAD_WITH_ALPACA = False
PLOT_SC = True
script_dir = os.path.dirname(__file__)


#### loading data #######
runs_to_load = [434493,434494]+[434506,434507]+[43611,43612]+[434651,434652]+[434666]

if LOAD_WITH_ALPACA:
    # retrieve the data
    
    first_run = runs_to_load[0]
    last_run = runs_to_load[-1]
    bad_runs = [i for i in range(first_run,last_run) if i not in runs_to_load]
    print(bad_runs)

    SC56Clock_label = 'SC56_coinc*event_clock'
    HVScan_label = 'Batman*acq_0*HCI_HV_Scan'
    HV_label = 'Batman*acq_0*NegHV_Ch2'
    SyncCheckLabels_label = 'metadata*SyncCheck_labels'
    run_time_label = "run_dir_creation_time*run_dir_creation_time_s"


    data = finalize.generate(first_run=first_run, #426447,426472
                            last_run=last_run,#426467,426473 
                            elog_results_filename=f'HV_scan_{first_run}_{last_run}',
                            known_bad_runs=bad_runs,
                            verbosing=True,
                            variables_of_interest=[
                                'Run_Number*Run_Number*__value',  # run number
                                'script_name',
                                run_time_label,
                                HV_label,
                                HVScan_label,
                                'Sync_check',
                                SyncCheckLabels_label,
                                SC56Clock_label
                                ],
                                directories_to_flush=['bronze', 'gold','dataset','elog'],
                                speed_mode=True) #'bronze', 'gold','dataset', 'elog'

    # change naming convension from loading to accessing
    run_time_label = run_time_label.replace('*','_')
    SC56Clock_label = SC56Clock_label.replace('*','_')
    HVScan_label = HVScan_label.replace('*','_')
    SyncCheckLabels_label = SyncCheckLabels_label.replace('*','_')
    HV_label = HV_label.replace('*','_')

    # condition data
    scintillators = pd.DataFrame(columns=['Run Number','t [s]'])
    sync_checks = pd.DataFrame(columns=['Run Number','clock','label'])
    run_params = pd.DataFrame(columns=['Run Number'])


    runs = data['Run_Number_Run_Number___value']
    print(runs)

    # sync checks
    for i,run_number in enumerate(runs):
        print("++++++++++++++++++++++++++++")
        print(run_number)
        print()

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
        # scintillators
        try:
            sc = data[SC56Clock_label][i]
            sc = pd.DataFrame({'Run Number':[run_number]*len(sc),'t [s]':sc})
            scintillators = pd.concat([scintillators,sc],ignore_index=True)
        except TypeError:
            print('WARNING: missing SC56 data')

        # run parameters 
        try:
            run_params = pd.concat([run_params,pd.DataFrame({'Run Number':run_number,'script_name':data['script_name'][i],'script_time':data[run_time_label][i],'HV_Scan':data[HVScan_label][i],'HV [V]':data[HV_label][i]},index=[run_number])])
        except TypeError:
            print('WARNING: some information about the run')  
    scintillators.to_csv(os.path.join(script_dir,'scintillators.csv'),index=False)
    sync_checks.to_csv(os.path.join(script_dir,'SyncChecks.csv'),index=False)
    run_params.to_csv(os.path.join(script_dir,'run_parameters.csv'),index=False)
    run_params = run_params.set_index('Run Number')
    data = None
else:
    sync_checks = pd.read_csv(os.path.join(script_dir,'SyncChecks.csv'))
    scintillators = pd.read_csv(os.path.join(script_dir,'scintillators.csv'))
    run_params = pd.read_csv(os.path.join(script_dir,'run_parameters.csv'),index_col='Run Number')


runs = run_params[run_params['script_name']=='HCI_CatchAndDump_v2.0.py'].index.to_list()
print(runs)


################ PLOTS #########################
plt.rcParams["figure.figsize"] = (2*6.4, 2*4.8)
# plt.tight_layout()



fig2 = plt.figure("sc")
ax_sc = fig2.subplots(len(runs),1,gridspec_kw = {'wspace':0.2, 'hspace':0.0},sharex='col') # gridspec_kw = {'wspace':0.2, 'hspace':0.0}
for i,run_number in enumerate(runs):
    sync_check_t = sync_checks[(sync_checks['Run Number']==run_number) & (sync_checks['label']=='NegHV_RampDownDegrader_start')]['clock'].to_list()
    sync_check_t.append(sync_check_t[-1]+np.mean([sync_check_t[j]-sync_check_t[j-1] for j in range(1,len(sync_check_t))]))
    scintillators.loc[scintillators['Run Number']==run_number, ['HV [V]']] = pd.cut(scintillators['t [s]'],sync_check_t,right=True,labels=[run_params['HV [V]'][run_number]-j for j in range(1,len(sync_check_t))])
    hv = scintillators.loc[(scintillators['Run Number']==run_number),['t [s]','HV [V]']].groupby('HV [V]').count()
    
    
    if PLOT_SC:
        ax_sc[i].fill_between([t-sync_check_t[0] for t in reversed(sync_check_t)],[0]+hv['t [s]'].to_list(),step='pre',color='tab:blue',alpha=0.4)
        ax_sc[i].plot([t-sync_check_t[0] for t in reversed(sync_check_t)],[0]+hv['t [s]'].to_list(),color='tab:blue',drawstyle='steps-pre')

        # plotting raw scintillator signal
        ax_sc[i].hist(scintillators[(scintillators['Run Number']==run_number) & (scintillators['t [s]'] <= sync_check_t[-1]) & (scintillators['t [s]'] >= sync_check_t[0])]['t [s]'] - sync_check_t[0],bins=400,label=run_number,color='tab:orange')
        ax_sc[i].set_xlabel('t [s]')
        ax_sc[i].set_ylabel('counts')


        for t in sync_check_t:
            ax_sc[i].axvline(t-sync_check_t[0],lw=1,color='red',alpha=0.5)
        ax_sc[i].xaxis.set_major_locator(MultipleLocator(10))
        ax_sc[i].xaxis.set_minor_locator(MultipleLocator(5))
        ax_sc[i].legend()
    
    

# file = os.path.join(os.path.split(__file__)[0],'plots',f'SC56_{run_number}.png')
# fig.savefig(file)
plt.show()

        