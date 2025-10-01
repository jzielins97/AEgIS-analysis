import unittest

import ALPACA.data.finalize as finalize
import ALPACA.configurations.experiment as experiment
import ALPACA.analyses.plot as plot
import ALPACA.configurations.verbose as verbose
import numpy as np

class SCHistogramPlot(unittest.TestCase):
    """
    This test should
    -create a histogram plot of the 
    """


    def test_sc_histogram_plot(self):
        """
        Creates the histograms of events observed via the SC56_coinc.
        """

        # 426397
        data = finalize.generate(first_run=432877 ,
                                 last_run=432883 ,
                                 elog_results_filename='run_viewer',
                                 known_bad_runs=[], #393842, 393844],
                                 verbosing=True,
                                 variables_of_interest=[
                                     'Run_Number*Run_Number*__value',  # run number
                                     'Batman*acq_0*HCI_HV_Scan',
                                     'run_dir_creation_time*run_dir_creation_time_str'
                                    ],
                                 directories_to_flush=['bronze', 'gold', 'datasets', 'elog'],
                                 speed_mode=False) #'bronze', 'gold', 'datasets', 'elog'

        print(data['run_dir_creation_time_run_dir_creation_time_str'])

if __name__ == '__main__':
    unittest.main()
