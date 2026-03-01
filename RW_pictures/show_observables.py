import unittest

import ALPACA.data.finalize as finalize
import ALPACA.configurations.experiment as experiment
import ALPACA.analyses.plot as plot
import ALPACA.configurations.verbose as verbose
import numpy as np
import polars as pl
import os

'''
I want to make a plot of rotating wall effects on the plasma.
There was a scan did by Benji and Rugg https://aegisgateway.cern.ch:8443/elog/RunLog/6293
486498 - initial run with no RW and a cold storage time of 0 seconds, establishing our initial conditions
486504-486515 Scan 1, RW Time from 10 to 120 s in steps of 10 s with 4.0 MHz and 0.3 V, no cold storage
486516-486527 Scan 2, RW Freq from 0.5 to 6.0 MHz step of 0.5 MHz and 0.3 V and 90s of time, no cold storage
486528-486539 Scan 3, RW Ampl from 0.05 to 0.65 V step of 0.05 V and 4.0 MHz and 90s of time, no cold storage
486540-486551 Scan 4, cold storage in dump trap after 90 s of RW at 4 MHz, 0.3 V, scanned from 15 to 180 s in steps of 15 s
And the second entry https://aegisgateway.cern.ch:8443/elog/RunLog/6311
RW scan from
487954 - 487963: 10 x RW time
487964 - 487975: 12 x RW Freq
487976 - 487987: 12 x RW Amp

'''
data = finalize.generate(first_run=486498,
                            last_run=486539,
                            elog_results_filename='run_viewer',
                            known_bad_runs=[486503],
                            verbosing=True,
                            variables_of_interest=[
                                'Run_Number*Run_Number*__value', # run number
                                'Batman*acq_0*Pos_RWTime_s',
                                'Batman*acq_0*Pos_RWFreq_MHz',
                                'Batman*acq_0*Pos_RWAmpl_V',
                                'PCOEdge*acq_1*height',
                                'PCOEdge*acq_1*width',
                                'PCOEdge*acq_1*C_flatten_data'
                            ],
                            directories_to_flush=["bronze","gold","datasets","elog"], # "bronze","gold","datasets","elog" 
                            speed_mode=True)
print(data)

df = pl.DataFrame({
    'run':data['Run_Number*Run_Number*__value'.replace('*','_')],
    'RW_time [s]':data['Batman*acq_0*Pos_RWTime_s'.replace('*','_')],
    'RW_freq [MHz]':data['Batman*acq_0*Pos_RWFreq_MHz'.replace('*','_')],
    'RW_ampl [V]':data['Batman*acq_0*Pos_RWAmpl_V'.replace('*','_')],
    'img_height':data['PCOEdge*acq_1*height'.replace('*','_')],
    'img_width':data['PCOEdge*acq_1*width'.replace('*','_')],
    'image':data['PCOEdge*acq_1*C_flatten_data'.replace('*','_')],
})
print(df)
df.write_parquet(os.path.join(os.path.dirname(__file__),"..","data","RW_scan.parquet"))