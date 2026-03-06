import ALPACA.data.finalize as finalize
import polars as pl
import numpy as np
import os
    
# retrieve the data
# 489710 - 489717 can with moving HH MCP (with uncalibrated )
# 489879 - 489891 
#

variables_of_interest = [
                            'Run_Number*Run_Number*__value',  # run number
                            'Batman*acq_0*HedgeHog_MCP_Degrees',
                            'Batman*acq_0*ELENA_H_OFFSET',
                            'Batman*acq_0*ELENA_V_OFFSET',
                            'Batman*acq_0*ELENA_H_ANGLE',
                            'Batman*acq_0*ELENA_V_ANGLE',
                            'Hikrobot*acq_0*height',
                            'Hikrobot*acq_0*width',
                            'Hikrobot*acq_0*C_flatten_data',
                            'ELENA_Parameters*H_offset_mm',
                            'ELENA_Parameters*V_offset_mm',
                            'ELENA_Parameters*H_angle_mrad',
                            'ELENA_Parameters*V_angle_mrad',
                        ]+[
                            f'ELENA_Parameters*H{id}_Corrector_{pos}' for id in [1,2,3] for pos in ['L','R']
                        ]+[
                            f'ELENA_Parameters*V{id}_Corrector_{pos}' for id in [1,2,3] for pos in ['T','B']    
                        ]+[
                            f'ELENA_Parameters*{dir}{id}_{pos}' for dir in ['QF','QD'] for id in [1,2] for pos in ['P','N']
                        ]
                        
column_names = [
    'run',
    'HH_MCP_position',
    'H_offset_requested',
    'V_offset_requested',
    'H_angle_requested',
    'V_angle_requested',
    'MCP_img_height',
    'MCP_img_width',
    'MCP_img',
    'H_offset_mm',
    'V_offset_mm',
    'H_angle_mrad',
    'V_angle_mrad',
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

failed_runs = []
# old scan offsets 492042:492482 angles: 492483:492651
# after fixing the beam steering
# small scan offsets 492673:492688
# small scan angles 492693:492709

filepath = os.path.join(os.path.dirname(__file__),"data","PbarsOnHHMCP_scan.parquet")

first_run = 492673
last_run = 492709
bad_runs = [idx for idx in range(492689,492693)] + [492708]
os.mkdir(os.path.join(os.path.dirname(__file__),"data","tmp_run_data"))
for run in range(first_run,last_run+1):
    if run in bad_runs:
        continue
    data = finalize.generate(first_run=run,
                            last_run= run, # 492482, # 489891
                            elog_results_filename='HHMCP',
                            known_bad_runs=[],
                            verbosing=False,
                            variables_of_interest=variables_of_interest,
                                directories_to_flush=['bronze', 'gold', 'datasets', 'elog'], #'bronze', 'gold', 'datasets', 'elog'
                                speed_mode=True) #'bronze', 'gold', 'datasets', 'elog'

            
    try:
        # print(data)
        bad_run = False
        for variable in variables_of_interest:
            if isinstance(data[variable.replace('*','_')][0],np.float64) and np.isnan(data[variable.replace('*','_')][0]):
                bad_run = True
                break
        if bad_run:
            failed_runs.append({run:'nan'})
            continue
        data = pl.DataFrame({col_name:data[variable.replace('*','_')] for col_name,variable in zip(column_names,variables_of_interest)})
        print(data)
        data.write_parquet(os.path.join(os.path.dirname(__file__),"data","tmp_run_data",f"PbarsOnHHMCP_run={run}.parquet"))
    except KeyError as e:
        print(e)
        failed_runs.append({run:'KeyError'})
    except TypeError as e:
        print(data['Run_Number*Run_Number*__value'.replace('*','_')])
        print(data['ELENA_Parameters*H_offset_mm'.replace('*','_')])
        print(data['ELENA_Parameters*V_offset_mm'.replace('*','_')])
        print(data['ELENA_Parameters*H_angle_mrad'.replace('*','_')])
        print(data['ELENA_Parameters*V_angle_mrad'.replace('*','_')])
        print(data['Batman*acq_0*HedgeHog_MCP_Degrees'.replace('*','_')])
        print(data['Hikrobot*acq_0*width'.replace('*','_')])
        print(data['Hikrobot*acq_0*height'.replace('*','_')])
        print(data['Hikrobot*acq_0*C_flatten_data'.replace('*','_')])
        print(e)
        failed_runs.append({run:'TypeError'})
    except Exception as e:
        failed_runs.append({run:f"{type(e)}"})
        print(e)
print(failed_runs)
pl.scan_parquet(os.path.join(os.path.dirname(__file__),"data","tmp_run_data","PbarsOnHHMCP_run=*.parquet")).sink_parquet(filepath)
try:
    os.rmdir(os.path.join(os.path.dirname(__file__),"data","tmp_run_data"))
except Exception as e:
    print(e)