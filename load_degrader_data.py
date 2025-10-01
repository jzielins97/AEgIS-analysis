import ALPACA.data.finalize as finalize
import polars as pl

data = finalize.generate(first_run=424159,
                         last_run=424344,
                         elog_results_filename='equationII', # 
                         known_bad_runs=[424187],
                         verbosing=False,
                         variables_of_interest=[
                             'Run_Number*Run_Number*__value',
                             'Beam_Intensity*acq_0*value',
                             'run_dir_creation_time*run_dir_creation_time_s',
                             '1TCMOS*acq_0*background_corrected*signal_sum',
                             '1TCMOS*acq_0*background_corrected*std_background',
                             'HV Negative Read*V_2',
                             'HV Negative Read*V_0'
                         ],
                         directories_to_flush=['bronze', 'gold', 'datasets', 'elog'], # 'bronze', 'gold', 'datasets', 'elog'
                         speed_mode=False)


data_parquet = pl.DataFrame({"Run":data['Run_Number_Run_Number___value'],
                             "beam_intensity":data['Beam_Intensity_acq_0_value'],
                             "signal":data['1TCMOS_acq_0_background_corrected_signal_sum'],
                             "signal_err":data['1TCMOS_acq_0_background_corrected_std_background'],
                             "HV3_kV":data['HV Negative Read_V_2'],
                             "HV1_kV":data['HV Negative Read_V_0']})

print(data_parquet)
data_parquet.write_parquet("data/degrader_MCP.parquet")
