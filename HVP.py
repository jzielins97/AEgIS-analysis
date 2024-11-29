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
        data = finalize.generate(first_run=434334 ,
                                 last_run=434334 ,
                                 elog_results_filename='run_viewer',
                                 known_bad_runs=[393838, 393840, 393842, 393844, 393846], #393842, 393844],
                                 verbosing=True,
                                 variables_of_interest=[
                                     'Run_Number*Run_Number*__value',  # run number
                                     'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*t',
                                     'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*V',
                                     'captorius1*acq_0*Channel_2_TOF_ions*Y_[V]*V',
                                     'captorius1*acq_0*Channel_3_TOF_ions*Y_[V]*V',
                                     'Sync_check',
                                    #  'metadata'
                                    'SC12_coinc*event_clock',
                                    'SC12_coinc*event'
                                    ],
                                 directories_to_flush=['bronze', 'gold', 'datasets', 'elog'],
                                 speed_mode=False) #'bronze', 'gold', 'datasets', 'elog'


if __name__ == '__main__':
    unittest.main()
