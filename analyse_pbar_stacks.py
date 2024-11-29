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
import datetime




"""
This test should
-create a histogram plot of the 
"""
# observables
# kMCP_V_array = 'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*V' # NestedTrap_Long_Dump Antiproton_Cold_Dump
# kMCP_t_array = 'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*t'
kSC56 = 'SC56_coinc*event_clock'
kSC56_big_rate = 'SC56_coinc*5T_pbar_accumulation*initial_bg_rate'
kTrigger_Catch1 = 'Sync_check*acq_0*Timestamp*clock'
kTrigger_Catch2 = 'Sync_check*acq_1*Timestamp*clock'
kTrigger_Dump1 = 'Sync_check*acq_3*Timestamp*clock'
kTrigger_Dump2 = 'Sync_check*acq_4*Timestamp*clock'
kADInjection = 'AD_Injection'
kELENA_Inj = 'ELENA_Inj_Trigger'
kELENA_Ext = 'ELENA_Ext_Trigger'
kELENA_Final = 'ELENA_Final_Ext'


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

def GetMissingRunNumbers(runs:list[int],bad_runs:int=None):
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


def load_single_run(run_number,bad_runs):
    # Load the data
    data = finalize.generate(first_run=run_number,
                                last_run=run_number,
                                elog_results_filename='SC_viewer',
                                known_bad_runs= [],
                                verbosing=True,
                                variables_of_interest=[
                                    'Run_Number*Run_Number*__value',  # run number
                                    'SC1112_coin*event',
                                    'SC1112_coin*event_clock',
                                    'ELENA_Final_Ext',
                                    'ELENA_Ext_Trigger',
                                    'ELENA_Inj_Trigger',
                                    'AD_Injection',
                                    'Batman*acq_0*ELENA_Stacking',
                                    'Batman*acq_0*Ecooling_RWampl',
                                    'Batman*acq_0*Ecooling_RWfreq'
                                ],
                                directories_to_flush=['datasets','elog']) #'bronze', 'gold', 'gold','datasets', 'elog'

    # check for bad runs
    print(f"Run_Number={data['Run_Number_Run_Number___value'][0]}")
    print()
    try:
        if str(data['Batman_acq_0_ELENA_Stacking']) == 'nan' or len(data['ELENA_Final_Ext'][0]) < int(data['Batman_acq_0_ELENA_Stacking'][0]) or str(data['SC1112_coin_event_clock']) == 'nan' or str(data['SC1112_coin_event']) == 'nan':
            print('Bad run!')
            bad_runs.append(run_number)
        else:
            print('Good run!')
    
        return data
    except:
        print('Bad run!')
        bad_runs.append(run_number)

if __name__ == '__main__':
    runs = [i for i in range(417454,417483)] # 417454,417483
    bad_runs = []
    for run in runs:
        load_single_run(run,bad_runs)

    print(bad_runs)
    # col_spec = [[{'type': 'xy'}]]
    # rows = len(data['Run_Number_Run_Number___value'])
    # cols = 1
    # subplot = make_subplots(
    #     rows=rows, cols=cols,
    #     specs=col_spec * rows,
    #     column_widths=[0.2],
    #     horizontal_spacing=0.04,
    #     vertical_spacing=0.1)  # row_heights
        
    # # plot the data
    # plot_names = []
    # for idx, run_number in enumerate(data['Run_Number_Run_Number___value']):
    #     plot_names.append(f"Run {run_number}")

    #     SC56 = plot.px.histogram(x = data[qol.to_obsv(self.kSC56)][idx], nbins=1000)
    #     for trace in SC56.data:
    #         subplot.add_trace(trace=trace,row=idx+1, col=1)

    #     triggers = data[qol.to_obsv(self.kADInjection)][idx]
    #     for acq in triggers.keys():
    #         value = triggers[acq]['Timestamp']['clock']/10000000
    #         if value >= (data[qol.to_obsv(self.kTrigger_Catch1)][idx]+data[qol.to_obsv(self.kTrigger_Catch2)][idx])/2/10000000-5:
    #             subplot.add_vline(x=value,line_dash="dash", line_color="firebrick", row=idx+1,col=1)
        
    #     triggers = data[qol.to_obsv(self.kELENA_Ext)][idx]
    #     for acq in triggers.keys():
    #         value = triggers[acq]['Timestamp']['clock']/10000000
    #         if value >= (data[qol.to_obsv(self.kTrigger_Catch1)][idx]+data[qol.to_obsv(self.kTrigger_Catch2)][idx])/2/10000000-5:
    #             subplot.add_vline(x=value,line_dash="dash", line_color="green", row=idx+1,col=1)

    # subplot.update_layout(title=dict(text='SC56', font=dict(size=25, family='Times New Roman')),
    #                     height=900, width=1800, plot_bgcolor='White', paper_bgcolor="White",
    #                     showlegend=True, margin={'t': 100, 'b': 85, 'l': 85, 'r': 4})
        

    # for row in range(rows):
    #     subplot.update_yaxes(title=dict(text='count'), row=row+1, col=1)
    #     subplot.update_xaxes(title=dict(text='t [s]'), row=row+1, col=1)
        
    # subplot.show()

