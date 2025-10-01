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

runs = [427000+x for x in [300, 302, 304, 306, 307, 308, 309, 316]] # 97, 316
# runs = [427000+x for x in [300, 302, 303, 304, 306, 307, 308, 309, 310, 312, 314, 315, 316]]
bad_runs = [i for i in range(runs[0],runs[-1]) if i not in runs]

data = finalize.generate(first_run=runs[0],
                         last_run=runs[-1],
                         elog_results_filename='HCI_SCs_2', # 
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
                             'Batman*acq_0*Catch_HotStorageTime',
                             'Batman*acq_0*NestedTrap_TrapFloor',
                             'Batman*acq_0*NestedTrap_Wall',
                             'Batman*acq_0*Pbar_CoolingTime',
                             'Batman*acq_0*HCI_IonAccumulationTime',
                             'Batman*acq_0*NegHV_Ch1',
                             'Batman*acq_0*NegHV_Ch2'
                         ],
                         directories_to_flush=[], # 'bronze', 'gold', 'datasets', 'elog'
                         speed_mode=False)

# print(data)
palette1 = sns.color_palette("tab10")
fig = plt.figure()
nrows = 4
ncols = 2*(int(np.ceil(len(runs)/nrows)))
axes = fig.subplots(nrows,ncols,gridspec_kw={'width_ratios': [1 if i %2 else 3 for i in range(ncols)]})

palette = sns.color_palette("hls",10)
fig2 = plt.figure()
ax2 = fig2.subplots(4,2)
# SC56_bcgs = np.array([38.36500620362147,36.504216618107314,38.60063839826681,40.965607394941834,35.85330214552532,36.48642173852988,38.068442491386556])
# SC56_bcg_avg, SC56_bcg_std = 37.8348049986256, 1.6141818205212635 # counts per second
for i,run in enumerate(data["Run_Number_Run_Number___value"]):
    print(f"============== {run} ==============")
    print(f"NegHV_Ch1={data['Batman_acq_0_NegHV_Ch1'][i]}")
    print(f"NegHV_Ch2={data['Batman_acq_0_NegHV_Ch2'][i]}")
    print(f"NestedTrap_TrapFloor={data['Batman_acq_0_NestedTrap_TrapFloor'][i]}")
    print(f"NestedTrap_Wall={data['Batman_acq_0_NestedTrap_Wall'][i]}")
    print(f"Catch_HotStorageTime={data['Batman_acq_0_Catch_HotStorageTime'][i]}")
    print(f"Pbar_CoolingTime={data['Batman_acq_0_Pbar_CoolingTime'][i]}")
    print(f"ion_accumultaion_tim={data['Batman_acq_0_Catch_HotStorageTime'][i] - data['Batman_acq_0_Pbar_CoolingTime'][i]}")

    t = np.array([stamp - 1e-5 for stamp in data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_t'][i]])
    mq = mq_from_tof(t)
    V = data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_V'][i]

    axes[i%4,2*(i//4)+0].plot(t,V,label=run,color=palette1[i])
    axes[i%4,2*(i//4)+0].grid()
    if i == 0:
        axes[i%4,2*(i//4)+0].vlines(np.sqrt(150/180)*6.493e-6,ymin=2e-3,ymax=10e-3,color=palette1[-2],label=r'$\bar{p}$')
        axes[i%4,2*(i//4)+0].vlines(8.594e-6,ymin=2e-3,ymax=10e-3,color=palette1[-1],label='peak 4')
    else:
        axes[i%4,2*(i//4)+0].vlines(np.sqrt(150/180)*6.493e-6,ymin=2e-3,ymax=10e-3,color=palette1[-2])
        axes[i%4,2*(i//4)+0].vlines(8.594e-6,ymin=2e-3,ymax=10e-3,color=palette1[-1])
    axes[i%4,2*(i//4)+0].set_xlim(0.4e-5,0.9e-5)
    axes[i%4,2*(i//4)+0].set_ylim(-0.01,0.01)

    sync_checks = [data['Sync_check'][i][f'acq_{j}']['Timestamp']['clock']/1e7 for j in range(len(data['Sync_check'][i]))]
    # offset = sync_checks[-2]
    pbar_arrival_1 = sync_checks[0]
    pbar_arrival_2 = sync_checks[1]
    print(f"pbar_arrival_diff={pbar_arrival_2-pbar_arrival_1}")
    offset = sync_checks[2]
    print(f"pbar_storage_time={offset - pbar_arrival_2}")

    sync_checks = [stamp - offset for stamp in sync_checks]
    SC56 = np.array([point - offset for point in data['SC56_coinc_event_clock'][i]])
    # SC56_bcg = SC56[SC56 < min(SC56) + 5] # calculate background rate as the rate for the first 5 seconds of the acquisition
    # print(f'SC56_bcg = {len(SC56_bcg) / (SC56_bcg[-1] - SC56_bcg[0])}')
    # print(min(SC56),pbar_arrival_1 - offset)
    # SC56 = SC56[SC56 > 0]
    # ax2[i%4,i//4].hist(x=SC56,range=(-1840,160),bins=2000)
    ax2[i%4,i//4].hist(x=SC56,range=(-40,160),bins=2000) # range=(-40,160),
    # pbar hot storage
    ax2[i%4,i//4].axvspan(sync_checks[1], sync_checks[2], alpha=0.5, color=palette[0],label="pbar storage")
    # ion accumulation
    ax2[i%4,i//4].axvspan(sync_checks[1]+data['Batman_acq_0_Pbar_CoolingTime'][i], sync_checks[2], alpha=0.5, color=palette[1],label="ion accumulation")
    # HV scan (12 lowering periods)
    # ax2[i%4,i//4].vlines(sync_checks[2:26],ymin=0,ymax=8000,color=palette[2])
    ax2[i%4,i//4].axvspan(sync_checks[2], sync_checks[25], alpha=0.5, color=palette[2],label="HV scan")
    # Reshape HCI for TOF
    ax2[i%4,i//4].vlines(sync_checks[26],ymin=0,ymax=8000,color=palette[4])
    # SR1 -> 4 s
    ax2[i%4,i//4].axvspan(sync_checks[26], sync_checks[26]+4, alpha=0.5, color=palette[5],label='SR1')
    # SR2 -> ~10 s
    ax2[i%4,i//4].axvspan(sync_checks[26]+4, sync_checks[27] - 6, alpha=0.5, color=palette[6],label='SR2')
    # SR3 -> 5 s
    ax2[i%4,i//4].axvspan(sync_checks[27] - 6, sync_checks[27]-1, alpha=0.5, color=palette[7],label='SR3')
    # ion cooling time -> 1 s
    ax2[i%4,i//4].axvspan(sync_checks[27] - 1, sync_checks[27], alpha=0.5, color=palette[8],label='ion cooling')
    # TOF
    # ax2[i%4,i//4].vlines(sync_checks[27],ymin=0,ymax=8000,color=palette[9])
    ax2[i%4,i//4].axvspan(sync_checks[27], sync_checks[27]+9e-5, alpha=0.5, color=palette[9],label='TOF')
    ax2[i%4,i//4].set_yscale("log")
    ax2[i%4,i//4].set_xlim(-40,160) # -40
    # ax2[i%4,i//4].legend()

    cmos = data['1TCMOS_acq_1_C_flatten_data'][i]
    # x_px = [x for _ in range(data['1TCMOS_acq_1_height'][i]) for x in range(data['1TCMOS_acq_1_width'][i])]
    # y_px = [y for y in range(data['1TCMOS_acq_1_height'][i]) for _ in range(data['1TCMOS_acq_1_width'][i])]
    cmos_2d = np.reshape(cmos,(data['1TCMOS_acq_1_height'][i],data['1TCMOS_acq_1_width'][i]))
    im = axes[i%4,2*(i//4)+1].imshow(cmos_2d) # norm=LogNorm(vmin=None, vmax=None)
    axes[i%4,2*(i//4)+1].axis('off')

fig.legend(loc="lower right",ncols=2)
# handles, labels = fig.get_legend_handles_labels()
# axes[-1,-2].legend(handles, labels, loc='center',ncols=2)
axes[-1,-2].axis('off')
axes[-1,-1].axis('off')

handles, labels = ax2[0,0].get_legend_handles_labels()
ax2[-1,-1].legend(handles, labels, loc='center',ncols=2)
ax2[-1,-1].axis('off')

plt.show()
# data_parquet = pl.DataFrame({"Run":data['Run_Number_Run_Number___value'],
#                              "beam_intensity":data['Beam_Intensity_acq_0_value'],
#                              "signal":data['1TCMOS_acq_0_background_corrected_signal_sum'],
#                              "signal_err":data['1TCMOS_acq_0_background_corrected_std_background'],
#                              "HV3_kV":data['HV Negative Read_V_2'],
#                              "HV1_kV":data['HV Negative Read_V_0']})

# print(data_parquet)
# data_parquet.write_parquet("data/degrader_MCP.parquet")
