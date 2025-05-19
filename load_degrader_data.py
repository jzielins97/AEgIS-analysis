import ALPACA.data.finalize as finalize

data = finalize.generate(first_run=424159,
                         last_run=424344,
                         elog_results_filename='equationII',
                         known_bad_runs=[424187],
                         verbosing=False,
                         variables_of_interest=[
                             'Run_NumberRun_Numbervalue',
                             'Beam_Intensity*value',
                             'run_dir_creation_timerun_dir_creation_time_s',
                             '1TCMOSacq_0background_correctedsignal_sum',
                             'HV Negative ReadV_2',
                             'HV Negative ReadV_0'
                         ],
                         directories_to_flush=['bronze', 'gold', 'datasets', 'elog'],
                         speed_mode=True)

print(data)