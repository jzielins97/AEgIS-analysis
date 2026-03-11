from ALPACA_wrapper import ALPACA_load_data
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


scintillators = [12,34,56]

variables_of_interest = [
    'labeled_SyncChecks*labels',
    'labeled_SyncChecks*timestamps_daq_clock_s',
    'SC12_coinc*events_after_hot_dump',
    'SC12_coinc*events_in_interval',
    'SC12_coinc*event_clock',
    'SC34_coinc*events_after_hot_dump',
    'SC34_coinc*events_in_interval',
    'SC34_coinc*event_clock',
    'SC56_coinc*events_after_hot_dump',
    'SC56_coinc*events_in_interval',
    'SC56_coinc*event_clock'  
]
                        
column_names = [
    # 'SyncCheck_label',
    # 'SyncCheck_timestamp_s',
    # 'SC12_events_after_hot_dump',
    # 'SC12_events_in_interval',
    # 'SC12_event_clock',
    # 'SC34_events_after_hot_dump',
    # 'SC34_events_in_interval',
    # 'SC34_event_clock',
    # 'SC56_events_after_hot_dump',
    # 'SC56_events_in_interval',
    # 'SC56_event_clock'  
]

dtype = {
    'labeled_SyncChecks*labels':pl.List,
    'labeled_SyncChecks*timestamps_daq_clock_s':pl.List,
}
for sc in scintillators:
    dtype[f"SC{sc}_coinc*event_clock"] = pl.List

data = ALPACA_load_data(
    runs = [493825], # [493710,493714,493717,493723,493730], # [493686],
    # first_run = 493560,
    # last_run = 493565,
    variables_of_interest=variables_of_interest,
    # directories_to_flush=[],
    verbosing=False,
    speed_mode=False,
    column_names=column_names,
    output_file_name="ELENA_optimiser_check.parquet",
    dtype=dtype,
    download_from_ALPACA='all',
    cleanup_at_exit=False)

print(data)

sns.set_palette('tab10')
plt.rcParams["figure.figsize"] = (2*6.4, 2*4.8)
runs = pl.Series(data.select("Run Number")).to_list()
fig,axes = plt.subplots(len(runs),1,squeeze=False)
for run,ax in zip(runs,axes.flatten()):
    print(f"Run #{run}")
    run_data = data.filter(pl.col("Run Number") == run)
    labels = run_data.select("labeled_SyncChecks*labels").item()
    sync_checks = np.array(run_data.select("labeled_SyncChecks*timestamps_daq_clock_s").item())
    for i,sc in enumerate(scintillators):
        sc_data = np.array(run_data.select(f"SC{sc}_coinc*event_clock").item())
        t_start = sc_data[0]
        t_end = sc_data[-1]

        # move all timestamps
        sc_data = sc_data - t_start
        if i == 0:
            sync_checks = sync_checks - t_start
        pbar_arrival = sync_checks[2]
        
        bckg_counts = sc_data[sc_data < pbar_arrival]
        sc_data = sc_data[sc_data > 0]
        bckg_rate = abs(len(bckg_counts)/(bckg_counts[-1]-bckg_counts[0]))

        bins = int((t_end-t_start)/0.001)
        if bins > 100000:
            bins = 1000

        print(f'\tfrom {t_start} to {t_end} -> {t_end-t_start} in {bins} bins')
        print(f"\tfirst sync check = {sync_checks[0]}")
        print(f"\tcouts={len(sc_data[sc_data[-1]-50 < sc_data])-50*bckg_rate} vs {run_data.select(f'SC{sc}_coinc*events_in_interval').item()}")
        
        # for label,time in zip(labels,sync_checks):
        #     print(label,time)
        # continue
        sc_plot = ax.hist(x=sc_data,bins=bins,histtype='step',label=f'SC{sc}')#,ax=ax[0]) #,weights='count'
        ax.grid(axis='x',which='major',linestyle = "dashed",linewidth = 0.5,alpha=0.8)
        ax.grid(axis='x',which='minor',linestyle = "dashed",linewidth = 0.5,alpha=0.5)
    ax.set_xlim(-10,110)
    # ax.set_ylim(0,300)

    x_min,x_max = ax.get_xlim()
    y_min,y_max = ax.get_ylim()
    
    close_sync_checks = 0
    for i,(clock,label) in enumerate(zip(sync_checks,labels)):
        if (clock < x_min) or (clock > x_max):
            continue
        last_position = 0
        # try:
        label = label
        ax.axvline(clock,lw=1,color='red',alpha=0.5)
        text_position = y_min+(0.8)*(y_max-y_min)
        if i >0 and (clock - sync_checks[i-1]) < 10:
            close_sync_checks += 1
            text_position -= close_sync_checks * 0.05 * (y_max-y_min)
        else:
            close_sync_checks = 0
        ax.text(clock,text_position,label,rotation=60,size='smaller')
    ax.axvspan(sc_data[-1]-50,sc_data[-1],color='gray',alpha=0.5)
    ax.legend()
    ax.set_yscale("log", nonpositive='mask')
    ax.set_xlabel("t [s]")
    ax.set_ylabel("scintillator counts")
    ax.set_title(f"{run}")
fig.tight_layout()

plt.show()