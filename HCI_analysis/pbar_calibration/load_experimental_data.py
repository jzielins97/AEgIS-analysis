import ALPACA.data.finalize as finalize
import polars as pl
import os
import matplotlib.pyplot as plt
import pickle

# first_run = 400395
# last_run = 400425
first_run = 434122
last_run = 434138
bad_runs = [434131]

data = finalize.generate(first_run=first_run,
                        last_run=last_run,
                        elog_results_filename='pbar_calibration',
                        known_bad_runs=bad_runs,
                        verbosing=False,
                        variables_of_interest=[
                            'Run_Number*Run_Number*__value',
                            # 'CinqueUnoCinqueDue_1TMCPPhosphor_Antiproton_Cold_Dump*acq_0*t', 
                            # 'CinqueUnoCinqueDue_1TMCPPhosphor_Antiproton_Cold_Dump*acq_0*V',
                            'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*t',
                            'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*V',
                            'Batman*acq_0*Ecooling_LaunchPotential', 
                            'Batman*acq_0*NestedTrap_TrapFloor', 
                            'Batman*acq_0*Catch_HotStorageTime', 
                            'Batman*acq_0*Catch_ColdStorageTime',       
                            'Batman*acq_0*Pbar_CoolingTime',
                            'Batman'
                        ],
                        directories_to_flush=['bronze', 'gold', 'datasets', 'elog' ],  # 'bronze', 'gold', 'datasets', 'elog' 
                        speed_mode=True)
runs = data['Run_Number_Run_Number___value']
for idx in range(len(runs)):
    print(runs[idx])
    print(type(data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_t'][idx]))


df = pl.DataFrame({
    'run':data['Run_Number_Run_Number___value'],
    'V_launch':data['Batman_acq_0_Ecooling_LaunchPotential'],
    'V_wall':[data['Batman'][i]['acq_0']['Ecooling_TrapEndcap'] for i in range(len(runs))],
    'HotStorage_s':data['Batman_acq_0_Catch_HotStorageTime'],
    'ColdStorage_s':data['Batman_acq_0_Catch_ColdStorageTime'],
    'CoolingTime_s':data['Batman_acq_0_Pbar_CoolingTime'],
    # 't_s':data['CinqueUnoCinqueDue_1TMCPPhosphor_Antiproton_Cold_Dump_acq_0_t'],
    # 'signal_V':data['CinqueUnoCinqueDue_1TMCPPhosphor_Antiproton_Cold_Dump_acq_0_V']
    't_s':data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_t'],
    'signal_V':data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_V']
})
df.write_parquet(os.path.join(os.path.dirname(__file__),'calibration_runs_part2.parquet'))
print(df)
'''


nrows = 5
ncols = len(runs) // nrows + 1
fig, ax = plt.subplots() # nrows,ncols,sharex=True,squeeze=True

for idx,run in enumerate(runs):
    print(run)
    try:
        print(data['Batman_acq_0_Ecooling_LaunchPotential'][idx])
        print(data['Batman_acq_0_NestedTrap_TrapFloor'][idx])
        print(data['Batman_acq_0_Catch_HotStorageTime'][idx])
        print(data['Batman_acq_0_Catch_ColdStorageTime'][idx])
        print(data['Batman_acq_0_Pbar_CoolingTime'][idx])
        t = data['CinqueUnoCinqueDue_1TMCPPhosphor_Antiproton_Cold_Dump_acq_0_t'][idx]
        V = data['CinqueUnoCinqueDue_1TMCPPhosphor_Antiproton_Cold_Dump_acq_0_V'][idx]
        print('Cold dump')
        print(t)
        print(V)
        # axs[idx%nrows][idx//nrows].plot(t,V,label=run)
        ax.plot(t,V,label=run)
    except Exception as e:
        print(e)
        bad_runs.append(run)
fig.legend()
fig.tight_layout()
plt.show()
'''
