from ctypes import alignment
import ALPACA.data.finalize as finalize

import os
from datetime import datetime

import pandas as pd
import numpy as np 
from scipy.optimize import curve_fit

# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
# from matplotlib.colors import LogNorm
  

LOAD_ALL_WITH_ALPACA = False
PLOT_SC = True
SCRIPT_DIR = os.path.dirname(__file__)
LIST_OF_RUNS = [434508,434509]+[434529,434530]+[434559,434560]+[434572,434573]+[i for i in range(434584,434606)] # ,434525,434526
# LIST_OF_RUNS = [434506, 434507, 434667, 434668, 434611, 434612, 434651, 434652, 434493, 434494, 434674, 434675]

def model(x,x0,sigma,a):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


try:
    run_params = pd.read_csv(os.path.join(SCRIPT_DIR,'run_parameters2.csv'))
    if LOAD_ALL_WITH_ALPACA:
        runs_to_load = LIST_OF_RUNS #  [run for run in LIST_OF_RUNS if run not in runs]
        run_params.drop(labels=run_params.index, axis=0, inplace=True)
    else:
        runs_to_load = [run for run in LIST_OF_RUNS if run not in run_params['Run Number'].to_list()]
    
except OSError:
    # file not found, create the dataframe
    run_params = pd.DataFrame(columns=['Run Number'])
    runs_to_load = LIST_OF_RUNS


########### loading data #######################
try:
    if LOAD_ALL_WITH_ALPACA:
        sync_checks = pd.DataFrame()
    else:
        sync_checks = pd.read_csv(os.path.join(SCRIPT_DIR,'SyncChecks2.csv'))
except OSError:
    sync_checks = pd.DataFrame()

try:
    if LOAD_ALL_WITH_ALPACA:
        scintillators = pd.DataFrame()
    else:
        scintillators = pd.read_csv(os.path.join(SCRIPT_DIR,'scintillators2.csv'))
except OSError:
    scintillators = pd.DataFrame()

if len(runs_to_load)>0:
    print("Missing runs:",runs_to_load)
    runs_to_load.sort()
    # retrieve the data
    first_run = runs_to_load[0]
    last_run = runs_to_load[-1]
    bad_runs = [i for i in range(first_run,last_run) if i not in runs_to_load]
    print(first_run)
    print(last_run)
    print(bad_runs)

    SC56Clock_label = 'SC56_coinc*event_clock'
    HVScan_label = 'Batman*acq_0*HCI_HV_Scan'
    HV_label = 'Batman*acq_0*NegHV_Ch2'
    HotStorageTime_label = 'Batman*acq_0*Catch_HotStorageTime'
    StorageTime_label = "Batman*acq_0*Pbar_StorageTime"
    SyncCheckLabels_label = 'metadata*SyncCheck_labels'
    run_time_label = "run_dir_creation_time*run_dir_creation_time_s"
    run_time_str_label = "run_dir_creation_time*run_dir_creation_time_str"


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
                                SC56Clock_label,
                                run_time_str_label,
                                HotStorageTime_label,
                                StorageTime_label
                                ],
                                directories_to_flush=['bronze', 'gold','elog'],
                                speed_mode=True) #'bronze', 'gold', 'elog'

    # change naming convension from loading to accessing
    run_time_label = run_time_label.replace('*','_')
    run_time_str_label = run_time_str_label.replace('*','_')
    SC56Clock_label = SC56Clock_label.replace('*','_')
    HVScan_label = HVScan_label.replace('*','_')
    SyncCheckLabels_label = SyncCheckLabels_label.replace('*','_')
    HV_label = HV_label.replace('*','_')
    HotStorageTime_label = HotStorageTime_label.replace('*','_')
    StorageTime_label = StorageTime_label.replace('*','_')

    runs = data['Run_Number_Run_Number___value']
    print(runs)

    # sync checks
    for i,run_number in enumerate(runs):
        print("++++++++++++++++++++++++++++")
        print(f'{run_number} ({data[run_time_str_label][i]})')
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
            run_params = pd.concat([run_params,pd.DataFrame({'Run Number':run_number,
                                                             'script_name':data['script_name'][i],
                                                             'script_time':data[run_time_label][i],
                                                             'script_time_str':data[run_time_str_label][i],
                                                             'HV_Scan':bool(data[HVScan_label][i]),
                                                             'HV [V]':data[HV_label][i],
                                                             'HotStorageTime':data[HotStorageTime_label][i],
                                                             'StorageTime':data[StorageTime_label][i]},
                                                             index=[run_number])])
        except TypeError:
            print('WARNING: some information about the run')  
    scintillators.to_csv(os.path.join(SCRIPT_DIR,'scintillators2.csv'),index=False)
    sync_checks.to_csv(os.path.join(SCRIPT_DIR,'SyncChecks2.csv'),index=False)
    run_params.to_csv(os.path.join(SCRIPT_DIR,'run_parameters2.csv'),index=False)
    data = None
# all data loaded 
# ################################################


run_params = run_params.set_index('Run Number')
runs = [run for run in run_params[(run_params['script_name']=='HCI_CatchAndDump.py') & (run_params['HV_Scan']) & (run_params['HotStorageTime']==180)].sort_values('script_time',ascending=True).index.to_list() if run in LIST_OF_RUNS]

run_time = []
peak_position = []
peak_position_err = []

################ PLOTS #########################
plt.rcParams["figure.figsize"] = (2*6.4, 2*4.8)
rows = 4
cols = int(np.ceil(len(runs)/rows))
if PLOT_SC:
    fig2 = plt.figure("sc")
    fig2.tight_layout()
    # ax_sc = fig2.subplots(len(runs),1,gridspec_kw = {'wspace':0.2, 'hspace':0.0},sharex='col') # gridspec_kw = {'wspace':0.2, 'hspace':0.0}
    ax_sc = fig2.subplots(rows,cols,gridspec_kw = {'wspace':0.05, 'hspace':0.0},sharex='col',sharey='row') 
for i,run_number in enumerate(runs):
    sync_check_t = sync_checks[(sync_checks['Run Number']==run_number) & (sync_checks['label']=='NegHV_RampDownDegrader_start')]['clock'].to_list()
    print(sync_check_t)
    sync_check_t.append(sync_check_t[-1]+np.mean([sync_check_t[j]-sync_check_t[j-1] for j in range(1,len(sync_check_t))]))
    scintillators.loc[scintillators['Run Number']==run_number, ['HV [V]']] = pd.cut(scintillators['t [s]'],sync_check_t,right=True,labels=[run_params['HV [V]'][run_number]-j for j in range(1,len(sync_check_t))])
    hv = scintillators.loc[(scintillators['Run Number']==run_number),['t [s]','HV [V]']].groupby('HV [V]').count().rename({'t [s]':'counts'},axis='columns')

    popt,pcov = curve_fit(model,hv.index,hv['counts'])

    print(f"Fiting results of {run_number}:")
    print(popt)
    (mean,sigma,scale) = popt
    print(pcov)
    print("=================================")

    script_time = datetime.strptime(run_params['script_time_str'][run_number],'%Y-%m-%d %H:%M:%S.%f')
    run_time.append(script_time) # example time '2024-11-29 13:01:57.000000'
    peak_position.append(mean)
    # peak_position_err.append(sigma)
    peak_position_err.append(pcov[0,0])
    
    
    if PLOT_SC:
        ax_sc[i%rows,i//rows].fill_between([t-sync_check_t[0] for t in reversed(sync_check_t)],hv['counts'].to_list()+[0],step='post',color='tab:blue',alpha=0.4)
        ax_sc[i%rows,i//rows].plot([t-sync_check_t[0] for t in reversed(sync_check_t)],hv['counts'].to_list()+[0],color='tab:blue',drawstyle='steps-post',label='sum')

        # plotting raw scintillator signal
        ax_sc[i%rows,i//rows].hist(scintillators[(scintillators['Run Number']==run_number) & (scintillators['t [s]'] <= sync_check_t[-1]) & (scintillators['t [s]'] >= sync_check_t[0])]['t [s]'] - sync_check_t[0],bins=60,fill=True,label=script_time,color='tab:orange')
        ax_sc[i%rows,i//rows].set_xlabel('t [s]')
        ax_sc[i%rows,i//rows].set_ylabel('counts')

        for t in sync_check_t:
            ax_sc[i%rows,i//rows].axvline(t-sync_check_t[0],lw=1,color='red',alpha=0.5)
        # ax_sc[i%rows,i//rows].xaxis.set_major_locator(MultipleLocator(10))
        # ax_sc[i%rows,i//rows].xaxis.set_minor_locator(MultipleLocator(5))

        # plot fitted curve
        ax_sc[i%rows,i//rows].plot([sync_check_t[j]-sync_check_t[0] + (sync_check_t[j+1]-sync_check_t[j])/2 for j in reversed(range(0,len(sync_check_t)-1))], model(hv.index.to_list(),mean,sigma,scale),'.--',color='green',label=f'{mean:.3}'+r"$\pm\sigma$"+f'={sigma:.3}')

        if i%rows == 0:
            ax2 = ax_sc[i%rows,i//rows].twiny()
            ax2.set_xlim(ax_sc[i%rows,i//rows].get_xlim())
            ax2.set_xticks([sync_check_t[t] - sync_check_t[0] for t in range(0,len(sync_check_t),2)])
            hv_labels = hv.index.to_list()+[12]
            ax2.set_xticklabels(int(hv_labels[j]) for j in range(len(hv_labels)-1,-1,-2))
            # ax2.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
            ax2.set_xlabel('HV [V]')
            # ax2.set_xticks([t - sync_check_t[0] for t in sync_check_t])
            # ax2.set_xticklabels(reversed(hv.index.to_list()+[12]))

            # ax2.set_xlabel("HV [V]")
        ax_sc[i%rows,i//rows].legend()
    
    
fig = plt.figure("time evolution")
fig.tight_layout()
ax = fig.subplots()
ax.errorbar(run_time,peak_position,yerr=peak_position_err,fmt='o',capthick=2,barsabove=True)
# ax.plot(run_time,peak_position,'o',color='tab:blue')
ax.set_xlabel('script time')
ax.set_ylabel('energy [keV]')
ax.tick_params(axis='x', labelrotation=45)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

y_min,y_max = ax.get_ylim()
ax.axvline(datetime.strptime("2024-11-29 23:15:00.000000",'%Y-%m-%d %H:%M:%S.%f'), lw=1,color='red',alpha=0.5)
ax.text(datetime.strptime("2024-11-29 23:15:00.000000",'%Y-%m-%d %H:%M:%S.%f'),y_max,"Pressure increased in reservoir to 30mbar and in CC UHV to 9.95e-7 mbar",va='top',rotation=-90)


# file = os.path.join(os.path.split(__file__)[0],'plots',f'SC56_{run_number}.png')
# fig.savefig(file)
plt.show()     