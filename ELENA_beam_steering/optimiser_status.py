from ALPACA_wrapper import ALPACA_load_data
import polars as pl

pl.Config.set_tbl_cols(20)

variables_of_interest = [
    'Batman*acq_0*ELENA_H_OFFSET',
    'Batman*acq_0*ELENA_V_OFFSET',
    'Batman*acq_0*ELENA_H_ANGLE',
    'Batman*acq_0*ELENA_V_ANGLE',
    'ELENA_Parameters*H_offset_mm',
    'ELENA_Parameters*V_offset_mm',
    'ELENA_Parameters*H_angle_mrad',
    'ELENA_Parameters*V_angle_mrad',
    'SC12_coinc*events_after_hot_dump',
    'SC12_coinc*event_clock',
    'SC12_coinc*events_in_interval'
]+[
    f'ELENA_Parameters*H{id}_Corrector_{pos}' for id in [1,2,3] for pos in ['L','R']
]+[
    f'ELENA_Parameters*V{id}_Corrector_{pos}' for id in [1,2,3] for pos in ['T','B']    
]+[
    f'ELENA_Parameters*{dir}{id}_{pos}' for dir in ['QF','QD'] for id in [1,2] for pos in ['P','N']
]
                        
column_names = [
    'H_offset_requested',
    'V_offset_requested',
    'H_angle_requested',
    'V_angle_requested',
    'H_offset_mm',
    'V_offset_mm',
    'H_angle_mrad',
    'V_angle_mrad',
    'SC12_coinc*events_after_hot_dump',
    'SC12_coinc*event_clock',
    'SC12_coinc*events_in_interval',
    'DHZE05L',
    'DHZE05R',
    'DHZE08L',
    'DHZE08R',
    'DHZE14L',
    'DHZE14R',
    'DVTE05T',
    'DVTE05B',
    'DVTE08T',
    'DVTE08B',
    'DVTE14T',
    'DVTE14B',
    'QFNE09P',
    'QFNE09N',
    'QFNE15P',
    'QFNE15N',
    'QDNE08P',
    'QDNE08N',
    'QDNE14P',
    'QDNE14N'
]

correctors = [
    'DHZE05R',
    'DVTE05T',
    'DHZE08R',
    'DVTE08T',
    'DHZE14R',
    'DVTE14T',
    'QDNE08P',
    'QDNE14P',
    'QFNE09P',
    'QFNE15P',
    ]

data = ALPACA_load_data(
    runs = [run for run in range(492714,492846+1)] + [run for run in range(492892,492944+1)],
    # first_run=492714,
    # last_run=492846,
    variables_of_interest=variables_of_interest,
    # directories_to_flush=[],
    # speed_mode=False,
    column_names=column_names,
    output_file_name="ELENA_beam_steering_simple_optimiser.parquet",
    dtype={'SC12_coinc*event_clock':pl.List},
    download_from_ALPACA='all',
    cleanup_at_exit=True)

print(data)
# for key,value in data.schema.items():
#     print(f"{key}:{value}")
best_settings = data.filter(pl.col("SC12_coinc*events_after_hot_dump")==pl.col("SC12_coinc*events_after_hot_dump").max())
print(best_settings.select("Run Number",*correctors,"SC12_coinc*events_after_hot_dump"))
