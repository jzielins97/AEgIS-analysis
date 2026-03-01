import unittest

import ALPACA.data.finalize as finalize
import ALPACA.configurations.experiment as experiment
import ALPACA.analyses.plot as plot
import ALPACA.configurations.verbose as verbose
import numpy as np

# 426397
data = finalize.generate(first_run=492033,
                            last_run=492033,
                            elog_results_filename='run_viewer',
                            known_bad_runs=[],
                            verbosing=True,
                            variables_of_interest=[
                                'Run_Number*Run_Number*__value', # run number
                                'Hikrobot',
                                'Hikrobot_3*acq_0',
                                'Hikrobot_3*acq_0*height',
                                'Hikrobot_3*acq_0*width',
                                'Hikrobot_3*acq_0*C_flatten_data',
                            ],
                            directories_to_flush=[], # "bronze","gold","datasets","elog" 
                            speed_mode=False)


print(data['Hikrobot_3*acq_0*height'.replace("*","_")])
for run,value in zip(data['Run_Number*Run_Number*__value'.replace('*','_')],data['Hikrobot_3*acq_0*height'.replace("*","_")]):
    print(run,value)