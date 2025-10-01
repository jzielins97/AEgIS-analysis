import ALPACA.data.finalize as finalize
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

def mq_from_tof(tof, V_floor = 180):
    '''
    Function to convert TOF to m/q (in uma/e).
    '''
    t_ref       = 6.493e-6
    m_q_ref     = 1.007276
    V_ref       = 150
    mq = ((tof/t_ref)**2) * (V_floor/V_ref) * m_q_ref
    return mq

runs = [427000+x for x in [300, 302, 304, 306, 307, 308, 309]] # 97, 316
# runs = [427000+x for x in [300, 302, 303, 304, 306, 307, 308, 309, 310, 312, 314, 315, 316]]
bad_runs = [i for i in range(runs[0],runs[-1]) if i not in runs]

data = finalize.generate(first_run=runs[0],
                         last_run=runs[-1],
                         elog_results_filename='HCI_data_load', # 
                         known_bad_runs=bad_runs,
                         verbosing=False,
                         variables_of_interest=[
                             'Run_Number*Run_Number*__value',
                             'Sync_check',
                             'SC56_coinc*event_clock',
                             'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*V',
                             'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*t',
                             '1TCMOS*acq_1*C_flatten_data',
                             '1TCMOS*acq_1*height',
                             '1TCMOS*acq_1*width',
                            #  'Batman*acq_0',
                            #  "HV Negative Read*V_2",
                            #  "HV Negative Read*V_0"
                             'Batman*acq_0*Catch_HotStorageTime',
                             'Batman*acq_0*NestedTrap_TrapFloor',
                             'Batman*acq_0*NestedTrap_Wall',
                             'Batman*acq_0*Pbar_CoolingTime',
                             'Batman*acq_0*HCI_IonAccumulationTime',
                             'Batman*acq_0*NegHV_Ch1',
                             'Batman*acq_0*NegHV_Ch2'
                         ],
                         directories_to_flush=['bronze', 'gold', 'datasets', 'elog'], # 'bronze', 'gold', 'datasets', 'elog'
                         speed_mode=True)

print(data)
# palette1 = sns.color_palette("tab10")
# fig = plt.figure()
# nrows = 4
# ncols = 2*(int(np.ceil(len(runs)/nrows)))
# axes = fig.subplots(nrows,ncols,gridspec_kw={'width_ratios': [1 if i %2 else 3 for i in range(ncols)]})

# palette = sns.color_palette("hls",10)
# fig2 = plt.figure()
# ax2 = fig2.subplots(4,2)

sync_check_labels = ['pbar_arrival_1','pbar_arrival_2']
for i in range(12):
    sync_check_labels.append(f'lower_HV_start_{i}')
    sync_check_labels.append(f"lower_HV_end_{i}")
sync_check_labels.append('trap_reshape_for_TOF')
sync_check_labels.append('perform_TOF')

ion_accumulation_s = []
sync_checks_all = []
SC56_all = []
SC56_bcg_all = []
for i,run in enumerate(data["Run_Number_Run_Number___value"]):
    print(f"============== {run} ==============")
    print(f"NegHV_Ch1={data['Batman_acq_0_NegHV_Ch1'][i]}")
    print(f"NegHV_Ch2={data['Batman_acq_0_NegHV_Ch2'][i]}")
    print(f"NestedTrap_TrapFloor={data['Batman_acq_0_NestedTrap_TrapFloor'][i]}")
    print(f"NestedTrap_Wall={data['Batman_acq_0_NestedTrap_Wall'][i]}")
    print(f"Catch_HotStorageTime={data['Batman_acq_0_Catch_HotStorageTime'][i]}")
    print(f"Pbar_CoolingTime={data['Batman_acq_0_Pbar_CoolingTime'][i]}")
    print(f"ion_accumultaion_time={data['Batman_acq_0_Catch_HotStorageTime'][i] - data['Batman_acq_0_Pbar_CoolingTime'][i]}")
    ion_accumulation_s.append(data['Batman_acq_0_Catch_HotStorageTime'][i] - data['Batman_acq_0_Pbar_CoolingTime'][i])
    # print(f"HV3_kV={data['HV Negative Read_V_2']}")
    # print(f"HV1_kV={data['HV Negative Read_V_0']}")
    # for i,(key,value) in enumerate(data['Batman_acq_0'][i].items()):
    #     if i > 2:
    #         print(key,value)

    t = np.array([stamp - 1e-5 for stamp in data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_t'][i]])
    mq = mq_from_tof(t)
    V = data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_V'][i]

    sync_checks = [data['Sync_check'][i][f'acq_{j}']['Timestamp']['clock']/1e7 for j in range(len(data['Sync_check'][i]))]
    offset = sync_checks[2]
    sync_checks_all.append([stamp - offset for stamp in sync_checks])
    SC56 = np.array([point - offset for point in data['SC56_coinc_event_clock'][i]])
    SC56_all.append(SC56)
    SC56_bcg = SC56[SC56 < min(SC56) + 5] # calculate background rate as the rate for the first 5 seconds of the acquisition
    SC56_bcg_rate = len(SC56_bcg) / (SC56_bcg[-1] - SC56_bcg[0])
    SC56_bcg_all.append(SC56_bcg_rate)


data_parquet = pl.DataFrame({"Run":data['Run_Number_Run_Number___value'],
                             "NegHV1_kV":data['Batman_acq_0_NegHV_Ch1'],
                             "NegHV3_kV":data['Batman_acq_0_NegHV_Ch2'],
                             "trap_floor_V":data['Batman_acq_0_NestedTrap_TrapFloor'],
                             "hot_storage_s":data['Batman_acq_0_Catch_HotStorageTime'],
                             "pbar_cooling_s":data['Batman_acq_0_Pbar_CoolingTime'],
                             "ion_accumulation_s":ion_accumulation_s,
                             'sync_check_s':sync_checks_all,
                             "sync_checks_labels":[sync_check_labels]*len(data['Run_Number_Run_Number___value']),
                             "SC56":SC56_all,
                             "SC56_bcg":SC56_bcg_all})

print(data_parquet)
data_parquet.write_parquet("data/HCI_PhD.parquet")
