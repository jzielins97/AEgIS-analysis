import unittest

import ALPACA.data.finalize as finalize
import ALPACA.configurations.experiment as experiment
import ALPACA.analyses.plot as plot
import ALPACA.configurations.verbose as verbose
import numpy as np

class SeeObservables(unittest.TestCase):
    """
    This test should
    -create a histogram plot of the 
    """


    def test_sc_histogram_plot(self):
        """
        Creates the histograms of events observed via the SC56_coinc.
        """

        data = finalize.generate(first_run=405810,
                                 last_run=405816,
                                 elog_results_filename='test_histogram_plot',
                                 known_bad_runs=[393838, 393840, 393842, 393844, 393846], #393842, 393844],
                                 verbosing=True,
                                 variables_of_interest=[
                                     'Run_Number*Run_Number*__value',  # run number
                                    ],
                                 directories_to_flush=['datasets','elog']) #'bronze', 'gold', 'gold','datasets', 'elog'
        
        return None


if __name__ == '__main__':
    unittest.main()


