from multiprocessing.managers import DictProxy
from typing import Tuple

import ALPACA.data.finalize as finalize
import ALPACA.configurations.experiment as experiment
import ALPACA.analyses.plot as plot
import ALPACA.configurations.verbose as verbose

import qol_functions as qol

from plotly.subplots import make_subplots

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import re
from datetime import datetime

"""
Creates the histograms of events observed via the SC56_coinc.
"""
def GetRetryTimes():
    '''
    Based on the error log find all times of the general retry calls
    '''
    filename = 'Error_Log_2024-05-29_13-52-03.842.txt'
    retry_datetimes = []
    with open(filename,'r') as f:
        for i, line in enumerate(f):
            error_code = re.search(r'(Code:) (5801|7105|7109|5218)',line) # retries are: banana returned retry, beam stopper in, empty shot, daq error
            if error_code is None:
                continue
            error_code = error_code.group()
            datetime_string = re.search(r'\d+:\d+:\d+\.\d+ \d+/\d+/\d+',line).group()
            retry_datetimes.append(datetime.strptime(datetime_string,r'%H:%M:%S.%f %d/%m/%Y'))
    return retry_datetimes

def GetMissingRunNumbers(df,runs:list[int],bad_runs:int=None):
    '''
    Function appends values missing from the list of the good runs to the list of bad runs
    '''
    # create list of missing runs based on the list of runs provided
    missing_runs = [i for i in range(runs[0],runs[-1]+1) if i not in runs]
    if bad_runs is None:
        bad_runs = missing_runs
    else:
        bad_runs = bad_runs + missing_runs
    return bad_runs


def load_single_run(df, run_number,bad_runs,variables_of_interest):
    run_entry = {}
    # Load the data
    data = finalize.generate(first_run=run_number,
                                last_run=run_number,
                                elog_results_filename='SC_viewer',
                                known_bad_runs= [],
                                verbosing=True,
                                variables_of_interest=variables_of_interest,
                                directories_to_flush=['datasets','elog']) #'bronze', 'gold', 'gold','datasets', 'elog'

    # check for bad runs
    run_entry['run_number'] = int(data['Run_Number_Run_Number___value'][0])
    run_entry['creation_time'] = datetime.strptime(data['run_dir_creation_time_run_dir_creation_time_str'][0],r'%Y-%m-%d %H:%M:%S.%f')
    print(f"Run_Number={data['Run_Number_Run_Number___value'][0]}")
    print()
    is_bad_run = False
    for variable in variables_of_interest:
        variable_exists = str(data[variable.replace('*','_')]) != 'nan'  
        run_entry[variable.replace('*','_')] =  variable_exists
        if not variable_exists:
            is_bad_run = True
              
    
    if is_bad_run:
        print('Bad run!')
        bad_runs.append(run_number)
    else:
        print('Bad run!')
    df.loc[len(df.index)] = run_entry
    return data

if __name__ == '__main__':
    runs = [405994] # [i for i in range(417454,417483)] # 417454,417483
    bad_runs = [405994]
    variables_of_interest = ['Run_Number*Run_Number*__value',  # run number
                             'run_dir_creation_time*run_dir_creation_time_str',
                             'SC1112_coin*event',
                             'SC1112_coin*event_clock',
                             'ELENA_Final_Ext',
                             'ELENA_Ext_Trigger',
                             'ELENA_Inj_Trigger',
                             'AD_Injection',
                             'Batman*acq_0*ELENA_Stacking',
                             'Batman*acq_0*Ecooling_RWampl',
                             'Batman*acq_0*Ecooling_RWfreq']
    columns = ['run_number','creation_time'] + [variable.replace('*','_') for variable in variables_of_interest]
    df = pd.DataFrame(columns=columns)

    for run in runs:
        load_single_run(df,run,bad_runs,variables_of_interest)
    
    print(df)

    print(bad_runs)

