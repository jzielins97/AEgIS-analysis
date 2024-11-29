import unittest
import numpy as np
import os
import ALPACA.data.finalize as finalize
import ALPACA.analyses.statistics as stats
import ALPACA.analyses.plot as plot
import ALPACA.configurations.hardware as hardware
import ALPACA.analyses.mathfunctions as mathfunctions
from tqdm import tqdm
import ALPACA.configurations.verbose as verbose
import ALPACA.configurations.paths as paths
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px


class BeamSteeringSubplots(unittest.TestCase):
    """
    This test should create a dashboard like plotly subplot instance for the beamsteering
    """

    def test_single_chart_analysis(self):
        """

        """

        data = finalize.generate(first_run=397457,
                                 last_run=397538,
                                 elog_results_filename='HCI_5814_all',
                                 verbosing=False,
                                 known_bad_runs=[],
                                 variables_of_interest=variables_of_interest,
                                 directories_to_flush=[],  # ['bronze', 'gold', 'datasets', 'elog'],
                                 speed_mode=False)

        # get the bad runs
        bad_runs = []
        for run in range(len(data['Run_Number_Run_Number___value'])):
            if str(data['CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_t'][run]) == 'nan':
                bad_runs.append(data['Run_Number_Run_Number___value'][run])
        print('bad_runs: ', bad_runs)
        print(len(data['Run_Number_Run_Number___value']))
        # slice as images get to big therefore data has to be cut
        data_copy = data.copy()

        # select runs
        for key in data_copy.keys():
            data.update({key: data_copy[key][:]})

        #peaks_t = stats.get_property_from_peaks_dict(peaks=data['CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_peaks'], property='t')
        #print(peaks_t)

        #print(np.diff(peaks_t))
        #integrals = stats.get_property_from_peaks_dict(peaks=data['CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_peaks'], property='integral', peak_number=0)
        #t_peak = stats.get_property_from_peaks_dict(peaks=data['CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_peaks'], property='t', peak_number=0)
        my_box_plot = plot.Chart(
            title=f'number of peaks detected ({data["Run_Number_Run_Number___value"][0]}_{data["Run_Number_Run_Number___value"][-1]})',
            my_run_list=[1, 2],
            x_axis_title=r"Trap Floor / V ",  # usage of Latex
            y_axis_title="peaks detected / count",
            chart_type="scatter",
            z_axis_title=None,
            zx_axis_title=None,
            x_data=np.array(data['Batman_acq_0_NestedTrap_TrapFloor']),
            y_data=np.array([len(data['CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_peaks'][idx]) for idx in range(len(data['Run_Number_Run_Number___value']))]),
            z_data=None,
            zx_data=None,
            zy_data=None,
            fit_params_first_guess=[1],
            data_fit_fcts={"y": None, "z": None},  # note: you are passing the reference not the instance.
            fit_error_bands=False,
            fit_error_band_opt='mse',  # mse, rmse, R2_score
            fit_confidence_intervals=False,
            filters={"x": None,  # lambda data: data[0] < 1400 and data[0] > 1100,
                     "y": None,  # lambda data: data[1] > 1022,
                     "z": None},
            show_chart=True,
            save_chart=True,
            add_custom_annotations=False)
        my_box_plot.make_chart()


        for idx in range(len(data[f'1TCMOS_acq_1_background_corrected_rings_avg'])):

            data[f'1TCMOS_acq_1_background_corrected_rings_avg'][idx] = data[f'1TCMOS_acq_1_background_corrected_rings_avg'][idx][1:]

        print(np.mean(np.array(data[f'1TCMOS_acq_1_background_corrected_rings_avg'])))

        my_box_plot = plot.Chart(
            title=f'TOFs from peaks with 5 sigma deviation from background. ({data["Run_Number_Run_Number___value"][0]}_{data["Run_Number_Run_Number___value"][-1]})',
            my_run_list=[1, 2],
            x_axis_title=r"Barrier Height / V",  # usage of Latex
            y_axis_title="t / s ",
            chart_type="scatter",
            z_axis_title=None,
            zx_axis_title=None,
            x_data=np.array(list(range(len(data['1TCMOS_acq_1_background_corrected_rings_avg'][0]))))*15,
            y_data=np.sum(np.array(data[f'1TCMOS_acq_1_background_corrected_rings_avg']), axis=0),
            z_data=None,
            zx_data=None,
            zy_data=None,
            fit_params_first_guess=[1],
            data_fit_fcts={"y": None, "z": None},  # note: you are passing the reference not the instance.
            fit_error_bands=False,
            fit_error_band_opt='mse',  # mse, rmse, R2_score
            fit_confidence_intervals=False,
            filters={"x": None,  # lambda data: data[0] < 1400 and data[0] > 1100,
                     "y": None,  # lambda data: data[1] > 1022,
                     "z": None},
            show_chart=True,
            save_chart=True,
            add_custom_annotations=False)
        my_box_plot.make_chart()
        quit(0)
        return None

    def test_make_full_peaks_analyses(self):
        """

        """

        data = finalize.generate(first_run=397457,
                                 last_run=397538,
                                 elog_results_filename='HCI_5814_all',
                                 verbosing=False,
                                 known_bad_runs=[],
                                 variables_of_interest=variables_of_interest,
                                 directories_to_flush=[],  # ['bronze', 'gold', 'datasets', 'elog'],
                                 speed_mode=False)

        # get the bad runs
        bad_runs = []
        for run in range(len(data['Run_Number_Run_Number___value'])):
            if str(data['CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_t'][run]) == 'nan':
                bad_runs.append(data['Run_Number_Run_Number___value'][run])
        print('bad_runs: ', bad_runs)
        print(len(data['Run_Number_Run_Number___value']))
        # slice as images get to big therefore data has to be cut
        data_copy = data.copy()

        # select runs
        for key in data_copy.keys():
            data.update({key: data_copy[key][:]})


        x = 't'
        var_x = data['Batman_acq_0_BarrierHeight']
        #TOF_filter = (0.000025, 0.000027) # test peak
        #TOF_filter = (0.0000107, 0.000011)  # peak 1
        #TOF_filter = (0.000014, 0.000016)  # peak 2
        #TOF_filter = (0.000018, 0.000020)  # peak 3
        #TOF_filter = (0.000020, 0.000022)  # peak 4
        #TOF_filter = (0.000023, 0.00002408) # peak 5
        var = 'FWHM_(s)'
        unique_peaks = [(0.0000105, 0.000013), (0.000012, 0.0000135),
                           (0.000018, 0.000020), (0.000020, 0.000022),
                           (0.000023, 0.00002408)]

        TOF_filters_140V = [(0.00000914, 0.00001299), (0.0000355, 0.000040), (0.0000515, 0.0000565),
                            (0.0000675, 0.0000715), (0.0000825, 0.0000875), (0.0000976, 0.0000997)]

        all_TOF_filters = TOF_filters_140V
        all_associated_settings = []
        all_ion_cooling_times = []
        all_peaks_x = []
        N = []
        E = []
        dNdE = []
        all_peaks_observable = []
        run_numbers = []
        setting = 'Batman_acq_0_BarrierHeight' #'Batman_acq_0_NegHV_Ch2'
        for run in range(len(data["Run_Number_Run_Number___value"])):
            run_number = data["Run_Number_Run_Number___value"][run]
            integral_sum = 0
            for peak_number in range(30):
                run_numbers.append(run_number)
                try:
                    all_peaks_x.append(data[f'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_peaks_peak_{peak_number}_{x}'][run])
                    all_peaks_observable.append(data[f'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_peaks_peak_{peak_number}_{var}'][run])

                    if not np.isnan(all_peaks_observable[-1]):
                        integral_sum += data[f'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_peaks_peak_{peak_number}_{var}'][run]
                    all_associated_settings.extend([data[f'{setting}'][run]])
                    all_ion_cooling_times.extend([data[f'Batman_acq_0_NestedTrap_IonCoolingTime'][run]])
                except KeyError:
                    pass

            N.append(integral_sum)
            E.append(data[f'{setting}'][run])

        print(len(all_associated_settings))
        print(len(all_peaks_x))
        print(all_peaks_x)
        print(len(all_peaks_observable))
        print(np.around(np.diff(np.array(all_peaks_x))*1000000, 3))
        masks = []
        for idx in range(len(all_TOF_filters)):
            mask_0 = np.array(all_peaks_x) > all_TOF_filters[idx][0]
            mask_1 = np.array(all_peaks_x) < all_TOF_filters[idx][1]
            combined_mask = mask_0 & mask_1
            masks.append(combined_mask)

        unique_Es = set(E)
        average_E = []
        average_N = []
        for e in unique_Es:

            E_avg = []
            N_avg = []
            for idx in range(len(E)):
                if E[idx] == e:
                    E_avg.append(E[idx])
                    N_avg.append(N[idx])
            average_E.append(e)
            if len(N_avg) > 0:
                average_N.append(sum(N_avg)/len(N_avg))

        temp_mask = np.array(all_peaks_x) > 0.0000#4423
        temp_mask_BH = np.array(all_associated_settings) <= 150
        temp_mask_ion_cooling = np.array(all_ion_cooling_times) > 0.0
        comb = temp_mask_BH & temp_mask & temp_mask_ion_cooling
        dt = np.diff(np.array(all_peaks_x)[~np.isnan(np.array(all_peaks_x))])
        my_box_plot = plot.Chart(
            title=f'IonCoolingTime 1s MRTOF_dt run({data["Run_Number_Run_Number___value"][0]}-{data["Run_Number_Run_Number___value"][-1]})',
            annotation_data=np.around(dt*1000000, 2),
            my_run_list=[1, 2],
            x_axis_title=f"TOF / s",  # usage of Latex
            y_axis_title=f'Barrier Height / V',
            chart_type="scatter",
            z_axis_title=f"{var}",
            zx_axis_title=None,
            x_data=np.array(all_peaks_x), #np.array(all_associated_settings)[masks[0]],
            y_data=np.array(all_associated_settings), #np.array(all_peaks_observable)[masks[0]],
            z_data=None, #np.array(all_peaks_observable)[comb],
            zx_data=np.array(all_associated_settings), #None, # np.array(all_associated_settings)[comb],
            zy_data=None,
            fit_params_first_guess=[1,1,1],
            data_fit_fcts={"y": None, "z": None},  # note: you are passing the reference not the instance.
            fit_error_bands=False,
            fit_error_band_opt='mse',  # mse, rmse, R2_score
            fit_confidence_intervals=False,
            filters={"x": None,  # lambda data: data[0] < 1400 and data[0] > 1100,
                     "y": None,  # lambda data: data[1] > 1022,
                     "z": None}, # lambda data: data[2] < TOF_filter[1] and data[2] > TOF_filter[0]},
            show_chart=True,
            save_chart=False,
            add_custom_annotations=False, add_traces=False,
            traces=[{'type': 'scatter',
                        'x_data': np.array(all_peaks_x)[masks[0]],#np.array(data[f'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_peaks_peak_{idx}_{x}'][idrun]),
                        'y_data': np.array(all_peaks_observable)[masks[0]],#np.array(data[f'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_peaks_peak_{idx}_{var}'][idrun]),
                        'title': 'peak 1', 'color': 'Black'},
                    {'type': 'scatter',
                     'x_data': np.array(all_peaks_x)[masks[1]],
                     'y_data': np.array(all_peaks_observable)[masks[1]],
                     'title': 'peak 2', 'color': 'Yellow'},
                    {'type': 'scatter',
                        'x_data': np.array(all_peaks_x)[masks[2]],
                        'y_data': np.array(all_peaks_observable)[masks[2]],
                        'title': 'peak 3', 'color': 'Blue'},
                    {'type': 'scatter',
                        'x_data': np.array(all_peaks_x)[masks[3]],
                        'y_data': np.array(all_peaks_observable)[masks[3]],
                        'title': 'peak 4', 'color': 'Green'},
                    {'type': 'scatter',
                        'x_data': np.array(all_peaks_x)[masks[4]],
                        'y_data': np.array(all_peaks_observable)[masks[4]],
                        'title': 'peak 5', 'color': 'Grey'},
                    {'type': 'scatter',
                     'x_data': np.array(all_peaks_x)[masks[5]],
                     'y_data': np.array(all_peaks_observable)[masks[5]],
                     'title': 'peak 6', 'color': 'Orange'}

                    ])# for idx in range(5) for idrun in range(len(data['Batman_acq_0_BarrierHeight']))])

        my_box_plot.make_chart()

        return None

    def test_make_multiple_subplots(self):
        """
        Stacks run data as subplots.
        """

        data = finalize.generate(first_run=397457,
                                 last_run=397538,
                                 elog_results_filename='HCI_5814_all',
                                 verbosing=False,
                                 known_bad_runs=[],
                                 variables_of_interest=variables_of_interest,
                                 directories_to_flush=[],  # ['bronze', 'gold', 'datasets', 'elog'],
                                 speed_mode=False)

        # get the bad runs
        bad_runs = []
        for run in range(len(data['Run_Number_Run_Number___value'])):
            if str(data['CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_t'][run]) == 'nan':
                bad_runs.append(data['Run_Number_Run_Number___value'][run])
        print('bad_runs: ', bad_runs)
        print(len(data['Run_Number_Run_Number___value']))
        # slice as images get to big therefore data has to be cut
        data_copy = data.copy()

        # select runs
        for key in data_copy.keys():
            data.update({key: data_copy[key][:]})

        # get events in the boil off
        #SC56_events = [data['SC56_coinc_events_without_sharp_peaks'][idx] - data['SC56_coinc_avg_background'][idx]*len(data['SC56_coinc_event_clock'][idx]) for idx in range(0, len(data['Run_Number_Run_Number___value']))]
        #bounds =[(300, 347), (286, 332), (269, 314)]
        #for idrun in range(len(data['SC56_coinc_event_clock'])):
        #    count = 0
        #    for idx in range(len(data['SC56_coinc_event_clock'][idrun])):
        #        if data['SC56_coinc_event_clock'][idrun][idx] <= bounds[idrun][1] and data['SC56_coinc_event_clock'][idrun][idx] >= bounds[idrun][0]:
        #            count += 1
       #     print(f'SC56_events, run {data["Run_Number_Run_Number___value"][idrun]}: {count}')
        #    print(f'SC56_integral, run {data["Run_Number_Run_Number___value"][idrun]}: {count/(bounds[idrun][1] - bounds[idrun][0])}')

        #print(SC56_events)

        # get size of the subplot
        number_runs = len(data['Run_Number_Run_Number___value'])

        col_spec = [[{'type': 'xy'}, {'type': 'scatter'}, {'type': 'table'}]]
        fig = make_subplots(
            rows=number_runs, cols=3,
            subplot_titles=[f'SC56', '1TMCP', 'Configs and Observables'],
            specs=col_spec * number_runs,
            column_widths=[0.2, 0.4, 0.4],
            horizontal_spacing=0,
            vertical_spacing=0)  # row_heights
        fig.update_layout(height=240 * len(data['Run_Number_Run_Number___value']), width=1400)

        for idx in tqdm(range(0, number_runs, 1), desc="adding subplots"):
            # add x histogram
            fig.add_trace(trace=go.Histogram(x=data['SC56_coinc_event_clock'][idx], showlegend=False),
                          row=idx+1, col=1)

            # add MCP_5152 readout
            fig.add_trace(trace=go.Scatter(x=data['CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_t'][idx][::4],
                                           y=data['CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_V_rebased'][idx][::4],
                                           showlegend=False,
                                           mode='markers', marker=dict(color='Black', size=3)),
                          row=idx+1, col=2)

            # add MCP_6133 readout
            #fig.add_trace(trace=go.Scatter(x=data['SeiUnoTreTre_MCPIn_Beam_Counter_and_MCP_acq_1_t'][idx][:110],
            #                              y=data['SeiUnoTreTre_MCPIn_Beam_Counter_and_MCP_acq_1_V'][idx][:110],
            #                               showlegend=False,
            #                               mode='markers', marker=dict(color='DarkBlue', size=3)),
            #              row=idx + 1, col=3)

            # add CMOS x profile readout
            #fig.add_trace(trace=go.Scatter(x=np.array(list(range(len(data['1TCMOS_acq_1_background_corrected_xprofile'][idx])))),
            #                               y=np.array(data['1TCMOS_acq_1_background_corrected_xprofile'][idx]),
            #                               showlegend=False,
            #                               mode='markers', marker=dict(color='DarkBlue', size=3)),
            #              row=idx + 1, col=3)

            # add CMOS avg rings profile readout
            #fig.add_trace(
            #    trace=go.Scatter(x=np.array(list(range(len(data['1TCMOS_acq_1_background_corrected_rings_avg'][idx]))))*4,
            #                     y=np.array(data['1TCMOS_acq_1_background_corrected_rings_avg'][idx]),
            #                     showlegend=False,
            #                     mode='markers', marker=dict(color='DarkBlue', size=3)),
            #    row=idx + 1, col=3)


            # add observable table
            batman_keys = ['NestedTrap_TrapFloor', 'BarrierHeight', 'NestedTrap_MRTOF_Time',
                           'NegHV_Ch1', 'NegHV_Ch2', 'NestedTrap_SqueezedTrapType', 'Pbar_EvapTrim',
                           'NestedTrap_OpeningPulseDuration', 'NestedTrap_IonCoolingTime']

            left = []
            right = []
            for key in batman_keys:
                if str(data[f'Batman_acq_0_{key}'][idx]) != 'nan':
                    if key == 'Pbar_CoolingTime':
                        left.append('Pbar_CoolingTime')
                        right.append(str(data[f'Batman_acq_0_{key}'][idx]))
                        continue
                    left.append(key)
                    right.append(data[f'Batman_acq_0_{key}'][idx])

            #left.append('spectrum_quality')
            #right.append("{:.3}".format(data['CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_spectrum_quality'][idx]))

            #left.append('events_in_boiloff')
            #right.append("{:.3e}".format(data['SC56_coinc_events_in_boiloff'][idx]))

            left.append('run start')
            right.append(data['Start_run'][idx][0]['Start_run_t_0'])

            #left.append('pbars_in_boiloff')
            #right.append("{:.3e}".format(data['SC56_coinc_pbars_in_boiloff'][idx]))

            #left.append('mcp integral')
            #right.append("{:.3e}".format(mcp_integrals[idx]))

            left.append('elena beam intensity')
            right.append("{:.3e}".format(data['Beam_Intensity_value'][idx]))

            left.append('peaks detected')
            right.append(len(data['CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_peaks'][idx]))

            #left.append('mcp noise')
            #right.append("{:.3e}".format(data['CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_mean_background_deviation'][idx]))

            #left.append('peak/noise')
            #right.append("{:.3e}".format(
            #    data['CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_SNR'][idx]))

            fig.add_trace(trace=go.Table(header=dict(values=['Run number', str(data['Run_Number_Run_Number___value'][idx])]),
                                         cells=dict(values=[left, right])), row=idx + 1, col=3)

            # add cmos sliced array
            #contrast = 0.8
            #fig.add_trace(trace=go.Heatmap(z=np.reshape(np.array(data['1TCMOS_acq_1_background_corrected_ring_sliced_array'][idx]), (2048, 2048)),
            #                              zmin=0,
            #                              zmax=np.max(data['1TCMOS_acq_1_background_corrected_ring_sliced_array'][idx])/contrast,
            #                              colorscale='plasma', showscale=False, showlegend=False),
            #             row=idx+1, col=4)

            # add cmos image
            #contrast = 1.0
            #fig.add_trace(trace=go.Heatmap(z=np.reshape(np.array(data['1TCMOS_acq_1_background_corrected_background_normalised_img'][idx]), (2048, 2048)),
            #                               zmin=0,
            #                               zmax=np.max(data['1TCMOS_acq_1_background_corrected_background_normalised_img'][idx])/contrast,
            #                               colorscale='plasma', showscale=False, showlegend=False),
            #              row=idx+1, col=1)

            # set the axes style of the histogram
            x_0 = data['SC56_coinc_event_clock'][idx][0] + hardware.SC56.analysis_config['boiloff_start_after_injection_peak']
            x_1 = x_0 + hardware.SC56.analysis_config['boiloff_duration_s'] + 200

            fig.update_xaxes(range=[x_0-100, x_1+100], tickvals=[200, 250, 300, 350, 400, 450, 500],row=idx+1, col=1)
            fig.update_yaxes(range=[-5, 8000], tickvals=[0, 5000, 10000], row=idx+1, col=1)
            #fig.update_xaxes(range=[0, 450], tickvals=[50,100,150,200,250, 300, 350, 400], row=idx+1, col=3)

            # set the axes style of the mcp signal
            #fig.update_xaxes(ticklabelposition='inside', row=idx + 1, col=2)
            #fig.update_yaxes(ticklabelposition='inside', row=idx + 1, col=2)

            fig.update_layout(title_text=f'Runs: {data["Run_Number_Run_Number___value"][0]}_{data["Run_Number_Run_Number___value"][-1]}')

        # write the image
        #fig.write_image(paths.path_to_images + os.sep + f'HCI_5810_2_runs_{data["Run_Number_Run_Number___value"][0]}_{data["Run_Number_Run_Number___value"][-1]}_HeInjection.png')

        # show the subplot
        fig.show()

        return None

    def test_make_subplots_for_multistep_acquisitions(self):
        """
        Stacks the mcp data from the multistep acquisition as subplots.
        """

        data = finalize.generate(first_run=397515,
                                     last_run=397515,
                                     elog_results_filename='HCI_Multistep',
                                     verbosing=False,
                                     known_bad_runs=[],
                                     variables_of_interest=variables_of_interest,
                                     directories_to_flush=['bronze', 'gold', 'datasets', 'elog'],  # ['bronze', 'gold', 'datasets', 'elog'],
                                     speed_mode=True)

        # get the bad runs
        bad_runs = []
        for run in range(len(data['Run_Number_Run_Number___value'])):
            if str(data['CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_t'][run]) == 'nan':
                bad_runs.append(data['Run_Number_Run_Number___value'][run])
        print('bad_runs: ', bad_runs)
        print(len(data['Run_Number_Run_Number___value']))
        # slice as images get to big therefore data has to be cut
        data_copy = data.copy()

        # select runs
        for key in data_copy.keys():
            data.update({key: data_copy[key][:]})

        number_runs = len(data['Run_Number_Run_Number___value'])
        number_acquisitions = 15
        subplot_titles = [f'acq_{idx}' for idx in range(number_acquisitions)]
        col_spec = [[{'type': 'scatter'} for _ in range(number_acquisitions)]]

        fig = make_subplots(
            rows=number_runs, cols=number_acquisitions,
            subplot_titles=subplot_titles,
            row_titles=[str(data['Run_Number_Run_Number___value'][idx]) for idx in range(number_runs)],
            specs=col_spec * number_runs,
            column_widths=[1/number_acquisitions for _ in range(number_acquisitions)],
            horizontal_spacing=0,
            vertical_spacing=0)  # row_heights
        fig.update_layout(height=640 * len(data['Run_Number_Run_Number___value']), width=450*number_acquisitions)

        for idx in tqdm(range(0, number_runs, 1), desc="adding subplots"):
            for jdx in range(0, number_acquisitions):
                # batman_keys = ['NestedTrap_TrapFloor', 'BarrierHeight',
                #                'NestedTrap_OpeningPulseDuration', 'NestedTrap_IonCoolingTime']
                # annotation_string = ''
                # for key in batman_keys:
                #     annotation_string += f'{key}: {data[f"Batman_acq_{jdx}_{key}"][idx]} <br> '
                # print(annotation_string)
                try:

                    # add MCP_5152 readout
                    mcp_plot = go.Scatter(x=data[f'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_{jdx}_t'][idx][:35000][::2],
                                          y=data[f'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_{jdx}_V_rebased'][idx][:35000][::2],
                                          showlegend=False,
                                          mode='markers', marker=dict(color='Black', size=3))
                    #mcp_plot.(text=f'{data["Run_Number_Run_Number___value"][idx]}', x=0.05, y=0.1, showarrow=False)
                    #mcp_plot.update_layout(annotations=[{'x': 13000, 'y': -0.4, 'xanchor': 'right', 'yanchor': 'bottom', 'text': annotation_string, 'font': {'size':5} }])
                    fig.add_trace(
                        trace=mcp_plot,
                        row=idx + 1, col=jdx+1)
                except (KeyError, TypeError, IndexError) as e:
                    fig.add_trace(
                        trace=go.Scatter(
                            x=np.array([0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006]),
                            y=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                            showlegend=False,
                            mode='markers', marker=dict(color='Black', size=3)),
                        row=idx + 1, col=jdx + 1)

            fig.update_layout(
                title_text=f'Multistep with Barrier. Runs: {data["Run_Number_Run_Number___value"][0]}_{data["Run_Number_Run_Number___value"][-1]}')

        # write the image
        #fig.write_image(paths.path_to_images + os.sep + f'HCI_5814_all_runs_{data["Run_Number_Run_Number___value"][0]}_{data["Run_Number_Run_Number___value"][-1]}_zoom.png')

        # show the subplot
        fig.show()

        return None

    def test_make_plot_with_MCP_and_CMOS_image(self):
        """
        Generates a plot with the MCP data and the CMOS image for one acquisition.
        """

        data = finalize.generate(first_run=397457,
                                 last_run=397538,
                                 elog_results_filename='HCI_5814_all',
                                 verbosing=False,
                                 known_bad_runs=[],
                                 variables_of_interest=variables_of_interest,
                                 directories_to_flush=[],  # ['bronze', 'gold', 'datasets', 'elog'],
                                 speed_mode=False)

        # get the bad runs
        bad_runs = []
        for run in range(len(data['Run_Number_Run_Number___value'])):
            if str(data['CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_0_t'][run]) == 'nan':
                bad_runs.append(data['Run_Number_Run_Number___value'][run])
        print('bad_runs: ', bad_runs)
        print(len(data['Run_Number_Run_Number___value']))
        # slice as images get to big therefore data has to be cut
        data_copy = data.copy()

        # select runs
        for key in data_copy.keys():
            data.update({key: data_copy[key][:]})

        # get the charts
        try:
            mcp_fig = go.Scatter(x=data[f'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_{jdx}_t'][idx][:35000],
                                 y=data[f'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump_acq_{jdx}_V_rebased'][idx][
                                   :35000],
                                 showlegend=False,
                                 mode='markers', marker=dict(color='Black', size=3))
        except Exception:
            mcp_fig = go.Scatter(
                x=np.array([0.00001, 0.00002, 0.00003, 0.00004, 0.00005, 0.00006]),
                y=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
                showlegend=False,
                mode='markers', marker=dict(color='Black', size=3))

        try:
            # cmos image and downgrade resolution
            cmos_image = go.Heatmap(
                z=np.reshape(np.array(data['1TCMOS_acq_1_background_corrected_background_normalised_img'][idx]),
                             (2048, 2048))[::2, ::2][::2, ::2][::2, ::2][::2, ::2],
                zmin=0,
                zmax=np.max(data['1TCMOS_acq_1_background_corrected_background_normalised_img'][idx]) / 0.8,
                colorscale='plasma', showscale=False, showlegend=False, xaxis='x2', yaxis='y2')

        except Exception:
            cmos_image = go.Heatmap(z=np.reshape(np.array([0, 1, 2, 3], (2, 2)),
                                                 zmin=0,
                                                 zmax=3,
                                                 colorscale='plasma', showscale=False, showlegend=False, xaxis='x2',
                                                 yaxis='y2'))

        # stack them
        charts = [mcp_fig, cmos_image]

        layout = go.Layout(
            # setting y-axis position for chart 2
            xaxis2=dict(
                domain=[0.75, 1],
                anchor='y2'
            ),
            # setting y-axis position for chart 2
            yaxis2=dict(
                domain=[0.05, 0.45],
                anchor='x2'))

        single_fig = go.Figure(data=charts, layout=layout)
        single_fig.write_image(paths.path_to_images + os.sep + f'run_{idx}_acq_{jdx}.png')
        single_fig.show()

        return None


if __name__ == '__main__':
    unittest.main()



# make the variables_of_interest

multistep_acquisitions = []
for idx in range(15):
    multistep_acquisitions.append(f'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_{idx}*t')
    multistep_acquisitions.append(f'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_{idx}*V_rebased')
    multistep_acquisitions.append(f'Batman*acq_{idx}*NestedTrap_OpeningPulseDuration')
    multistep_acquisitions.append(f'Batman*acq_{idx}*Pbar_CoolingTime')
    multistep_acquisitions.append(f'Batman*acq_{idx}*NegHV_Ch1')
    multistep_acquisitions.append(f'Batman*acq_{idx}*NegHV_Ch2')
    multistep_acquisitions.append(f'Batman*acq_{idx}*Catch_HotStorageTime')
    multistep_acquisitions.append(f'Batman*acq_{idx}*NestedTrap_IonStorageTime')
    multistep_acquisitions.append(f'Batman*acq_{idx}*NestedTrap_TrapFloor')
    multistep_acquisitions.append(f'Batman*acq_{idx}*NestedTrap_SqueezeTime')
    multistep_acquisitions.append(f'Batman*acq_{idx}*NestedTrap_SqueezedTrapType')
    multistep_acquisitions.append(f'Batman*acq_{idx}*NestedTrap_MRTOF_Time')
    multistep_acquisitions.append(f'Batman*acq_{idx}*NestedTrap_IonCoolingTime')
    multistep_acquisitions.append(f'Batman*acq_{idx}*NestedTrap_SqueezeRaise')
    multistep_acquisitions.append(f'Batman*acq_{idx}*Pbar_EvapTrim')
    multistep_acquisitions.append(f'Batman*acq_{idx}*BarrierHeight')
    multistep_acquisitions.append(f'Batman*acq_{idx}*MCP1T_In')


variables_of_interest = ['Run_Number*Run_Number*__value',
                         'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*V_rebased',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*V',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*mean_background_deviation',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*SNR',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*spectrum_quality',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_0*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_0*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_0*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_0*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_1*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_1*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_1*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_1*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_2*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_2*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_2*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_2*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_3*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_3*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_3*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_3*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_4*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_4*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_4*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_4*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_5*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_5*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_5*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_5*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_6*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_6*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_6*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_6*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_7*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_7*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_7*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_7*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_8*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_8*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_8*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_8*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_9*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_9*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_9*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_9*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_10*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_10*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_10*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_10*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_11*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_11*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_11*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_11*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_12*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_12*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_12*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_12*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_13*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_13*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_13*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_13*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_14*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_14*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_14*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_14*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_15*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_15*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_15*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_15*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_16*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_16*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_16*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_16*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_17*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_17*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_17*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_17*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_18*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_18*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_18*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_18*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_19*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_19*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_19*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_19*FWHM_(s)',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_20*t',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_20*abs_peak_height',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_20*integral',
                        'CinqueUnoCinqueDue_1TMCPPhosphor_NestedTrap_Long_Dump*acq_0*peaks*peak_20*FWHM_(s)',

                        '1TCMOS*acq_1*background_corrected*xhist',
                        '1TCMOS*acq_1*background_corrected*yhist',
                        '1TCMOS*acq_0*background_corrected*background_normalised_img',
                        '1TCMOS*acq_1*background_corrected*background_normalised_img',
                        '1TCMOS*acq_2*background_corrected*background_normalised_img',
                        '1TCMOS*acq_1*background_corrected*ring_sliced_array',
                        '1TCMOS*acq_1*height',
                        '1TCMOS*acq_1*width',
                        '1TCMOS*acq_1*background_corrected*rings_avg',
                        '1TCMOS*acq_1*background_corrected*xprofile',
                        '1TCMOS*acq_1*background_corrected*yprofile',
                        'SeiUnoTreTre_MCPIn_Beam_Counter_and_MCP*acq_1*t',
                        'SeiUnoTreTre_MCPIn_Beam_Counter_and_MCP*acq_1*V',
                        'ELENA_Parameters*H_offset_mm',
                        'ELENA_Parameters*V_offset_mm',
                        'ELENA_Parameters*H_angle_mrad',
                        'ELENA_Parameters*V_angle_mrad',
                        'Batman*acq_0*NestedTrap_OpeningPulseDuration',
                        'Batman*acq_0*Pbar_CoolingTime',
                        'Batman*acq_0*NegHV_Ch1',
                        'Batman*acq_0*NegHV_Ch2',
                        'Batman*acq_0*Catch_HotStorageTime',
                        'Batman*acq_0*NestedTrap_IonStorageTime',
                        'Batman*acq_0*NestedTrap_TrapFloor',
                        'Batman*acq_0*NestedTrap_SqueezeTime',
                        'Batman*acq_0*NestedTrap_SqueezedTrapType',
                        'Batman*acq_0*NestedTrap_MRTOF_Time',
                        'Batman*acq_0*NestedTrap_IonCoolingTime',
                        'Batman*acq_0*NestedTrap_SqueezeRaise',
                        'Batman*acq_0*Pbar_EvapTrim',
                        'Batman*acq_0*BarrierHeight',
                        'Batman*acq_0*MCP1T_In',
                        'SC56_coinc*event_clock',
                        'SC56_coinc*events_without_sharp_peaks',
                        'SC56_coinc*events_in_interval',
                        'SC56_coinc*avg_background',
                        'SC56_coinc*events_in_boiloff',
                        'SC56_coinc*pbars_in_boiloff',
                        'Beam_Intensity*value',
                        'Start_run',
                        'metadata',
                        ]

variables_of_interest.extend(multistep_acquisitions)