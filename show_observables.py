import unittest

import ALPACA.data.finalize as finalize
import ALPACA.configurations.experiment as experiment
import ALPACA.analyses.plot as plot
import ALPACA.configurations.verbose as verbose
import numpy as np

# 426397
RUN=492712
data = finalize.generate(first_run=RUN,
                            last_run=RUN,
                            elog_results_filename='run_viewer',
                            known_bad_runs=[],
                            verbosing=True,
                            variables_of_interest=[
                                'Run_Number*Run_Number*__value', # run number
                            ],
                            directories_to_flush=["bronze","gold","datasets","elog" ], # "bronze","gold","datasets","elog" 
                            speed_mode=False)