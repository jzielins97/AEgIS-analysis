import ALPACA.data.finalize as finalize
import polars as pl
import numpy as np

'''
We don't appreciate a big difference in the numbers on the MCP, there is at most a 5% difference - negligible. We now schedule the full thin foil scan using DegradeLadderPassThroughScan.py.
It performs, per each of the foils at 7, 28.7, 50.5, 78.6 and 96 mm an HV scan with HV3 from 0 to 15 kV in 0.2 kV step (76 runs per setting). The whole scan will take 19 hours. 

First run: 370169
Last run: 370548
End time: 8:15

After the scan of the centers with the Passthrough, we now try to see the difference in trapping efficiency by scheduling a similar measure using DegraderLadderCatchandDump.py.
It performs, per each of the foils at 7, 28.7, 50.5, 78.6 and 96 mm a trapping scan with HV1&2 from 0 to 14 kV in 0.2 kV step (71 runs per setting). HotStorage time set to 10 s.  The whole scan will take 20-24 hours. 

First run: 370549
Start time: 9:48
Stop time: 00:57
Stop run: 370983

Re taking some empty shots: Catch&Dump on foil 3 (50.5) from 2 kV to 7.6 kV (still with a step of 0.2).
Start run: 371119
Start time: 12:18
Stop run : 371150
end time: 15:10

Also on foil 4 (28.7) from 1.2 to 2.0 and from 13.4 to 14.0, and on foil 2 (78.6) from 0 to 4.4 and from 10.4 to 14.0.
They are done with PbarCatch&Dump, except for a run in the middle done with DeraderLadderCatch&DumpScan, used to chane the foil position!
Start run: 371161
start time: 16:35
Stop run: 371215
stop time: 20:11
'''

def load_data(first_run:int,last_run:int,bad_runs:list):
    data = finalize.generate(first_run=first_run,
                            last_run=last_run,
                            elog_results_filename='degraderSC', # 
                            known_bad_runs=bad_runs,
                            verbosing=False,
                            variables_of_interest=[
                                'Run_Number*Run_Number*__value',
                                'Beam_Intensity*acq_0*value',
                                'run_dir_creation_time*run_dir_creation_time_s',
                                'HV Negative Read*V_2',
                                'HV Negative Read*V_0',
                                'SC56_coinc*events_in_interval',
                                'SC56_coinc*avg_background',
                                'SC56_coinc*pbars_in_interval',
                                'SC78_coinc*events_in_interval',
                                'SC78_coinc*avg_background',
                                'SC78_coinc*pbars_in_interval',
                                'SC1112_coinc*events_in_interval',
                                'SC1112_coinc*avg_background',
                                'SC1112_coinc*pbars_in_interval',
                                'SC1920_coinc*events_in_interval',
                                'SC1920_coinc*avg_background',
                                'SC1920_coinc*pbars_in_interval',
                            ],
                            directories_to_flush=['bronze', 'gold', 'datasets', 'elog'], # 'bronze', 'gold', 'datasets', 'elog'
                            speed_mode=False)

    data_parquet = pl.DataFrame({"Run":data['Run_Number_Run_Number___value'],
                                "HV3_kV":data['HV Negative Read_V_2'],
                                "HV1_kV":data['HV Negative Read_V_0'],
                                "SC56_count":data['SC56_coinc_events_in_interval'],
                                "SC56_bcg":data['SC56_coinc_avg_background'],
                                "SC56_pbars":data['SC56_coinc_pbars_in_interval'],
                                "SC78_count":data['SC78_coinc_events_in_interval'],
                                "SC78_bcg":data['SC78_coinc_avg_background'],
                                "SC78_pbars":data['SC78_coinc_pbars_in_interval'],
                                "SC1112_count":data['SC1112_coinc_events_in_interval'],
                                "SC1112_bcg":data['SC1112_coinc_avg_background'],
                                "SC1112_pbars":data['SC1112_coinc_pbars_in_interval'],
                                "SC1920_count":data['SC1920_coinc_events_in_interval'],
                                "SC1920_bcg":data['SC1920_coinc_avg_background'],
                                "SC1920_pbars":data['SC1920_coinc_pbars_in_interval']})
    
    print(data_parquet)
    data_parquet.write_parquet(f"data/degrader_SC_{first_run}-{last_run}.parquet")


if __name__ == "__main__":
    # first set is 370549-370983
    # second set is 371119-371150
    # runs = [i for i in range(370549,370984)] + [i for i in range(371119,371151)]
    runs = [i for i in range(370809,370984)] + [i for i in range(371119,371151)]
    bad_runs = np.arange(370984,371119)
    missed_runs = []
    for i in range(0,len(runs),20):
        print(i,runs[i:i+20])
        # print(runs[i],runs[i+19])
        try:
            load_data(runs[i],runs[i+19],bad_runs=bad_runs)
        except:
            missed_runs+=runs[i:i+20]
