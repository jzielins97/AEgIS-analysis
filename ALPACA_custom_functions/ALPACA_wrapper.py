import ALPACA.data.finalize as finalize
import polars as pl
import numpy as np
import logging
import inspect
import os
import sys

logging.basicConfig(
    format="%(levelname)s:%(message)s" 
)

_log = logging.getLogger(__name__)
_log.propagate = True
_log.setLevel(level=logging.INFO)

def ALPACA_load_data(
    runs:list|None=None,
    first_run:int|None=None,
    last_run:int|None=None,
    variables_of_interest:list|None=None,
    known_bad_runs:list|None=None,
    directories_to_flush:list|None=None,
    speed_mode:bool=True,
    verbosing=False,
    column_names:list|None=None,
    dtype:dict|None=None,
    download_from_ALPACA:str="missing",
    data_path:str="data",
    output_file_name:str|None=None,
    cleanup_at_exit:bool=True,
    )->pl.DataFrame:
    """Helper function for downloading data using ALPACA.
    This function will download runs in run-by-run fashion and save
    the data in a temporary directory. This temporary directory is removed
    upon the completion of data retrieval.

    Args:
        runs (list | None, optional):
            List of runs to dowload. Defaults to None.
        first_run (int | None, optional, ALPACA.data.finalize parameter):
            First run to download. Can be used instead of explicit list of runs to define range of runs with first_run[:last_run+1]. Is used when runs=None. Defaults to None.
        last_run (int | None, optional, ALPACA.data.finalize parameter):
            Last run to download. When first_run is specified and last_run is None, a single run will be downloaded. Defaults to None.
        variables_of_interest (list | None, optional, ALPACA.data.finalize parameter):
            List of variables to be included in the downalod. Defaults to ['Run_Number*Run_Number*__value'].
        known_bad_runs (list | None, optional, ALPACA.data.finalize parameter):
            List of runs to be skipped in the download. Defaults to [].
        directories_to_flush (list | None, optional, ALPACA.data.finalize parameter):
            Specifies, from which data source you want the pipeline to be executed from. There are 4 levels of processed data: bronze, gold, datasets, elog. Defaults to ["bronze", "gold", "datasets", "elog"].
        speed_mode (bool, optional, ALPACA.data.finalize parameter):
            Set speed_mode=True to omit saving the different levels pipelines and will generate the datasets using only relevant data for the observables. Defaults to True.
        verbosing (bool, optional, ALPACA.data.finalize parameter):
            Set verbosing level of ALPACA. When verbosing=True the json gui will show up for each run loaded. Defaults to False.
        column_names (list | None, optional):
            List of column names in the final DataFrame. Should match the length of the variables_of_interest. By default, use variables_of_interest as column names.
        dtype (dict | None, optional):
            Mapping of the casting of columns into specific dtype. As keys it uses the column_names and as values it uses the dtypes from polars. Usually needed when the data from ALPACA is an array of variable length, which should be casted into pl.List. By default, it doesn't cast. 
        download_from_ALPACA (str, optional):
            Select which runs should be downloaded using ALPACA. Accepted options are:
            - 'none': load local file
            - 'all': download all specified runs even if the data already exists
            - 'missing': download only files missing from the local parquet file with the DataFrame.
        data_path (str, optional):
            Path to the folder where to create the temporary data folder. It is also used as the base path for saving the final DataFrame, when requested. Defaults to "<script_path>/data".
        output_file_name (str | None, optional):
            Name of the file in the data_path directory where to save the final DataFrame. When set to None, the DataFrame isn't saved. Defaults to None.
        cleanup_at_exit (bool, optional):
            When set to true, all temporary files will be deleted at when the function finishes.

    Raises:
        ValueError: Raised when both runs and first_run parameters are 

    Returns:
        pl.DataFrame: data downloaded form ALPACA
    """
    _CALLER_DIR = os.path.dirname(sys._getframe(1).f_globals['__file__'])
    _log.debug(_CALLER_DIR)
    _func_name = inspect.stack()[0][3]
    
    if known_bad_runs is None:
        known_bad_runs = []  
        
    if runs is None:
        if first_run is None:
            _log.error(f"{_func_name}:Cannot download data from ALPACA without specifying either list of runs or the first run")
            raise ValueError(f"{_func_name}:Cannot download data from ALPACA without specifying either list of runs or the first run")
        elif last_run is None:
            last_run = first_run
        elif last_run < first_run:
            first_run, last_run = last_run, first_run
        runs = [idx for idx in range(first_run,last_run+1) if idx not in known_bad_runs]
    else:
        runs=sorted(runs)
        first_run=runs[0]
        last_run=runs[-1]    
    
    _log.debug(f"{_func_name}:first_run={first_run}")
    _log.debug(f"{_func_name}:last_run={last_run}")
    
    # create output data path
    if data_path == "data":
        data_path = os.path.join(_CALLER_DIR,"data")
    output_file_path = os.path.join(data_path,output_file_name)
    
    # load data if exists without downloading from ALPACA
    if download_from_ALPACA.lower() == 'none':
        try:
            data = pl.read_parquet(output_file_path)
            return data.filter(pl.col("Run Number").is_in(runs))
        except FileNotFoundError:
            _log.error(f"{_func_name}: couldn't find the data file {output_file_path}")
            return
    
    # set default values
    if not dtype:
        dtype = {}
    
    if directories_to_flush is None:
        directories_to_flush = ['bronze', 'gold', 'datasets', 'elog']
    
    if variables_of_interest is None:
        variables_of_interest = []
    
    tmp_directory_created = False
    if not os.path.exists(os.path.join(data_path,'tmp_run_data')):
        _log.info(f"{_func_name}:Creating directory at {os.path.join(data_path,'tmp_run_data')} for run data")
        os.mkdir(os.path.join(data_path,'tmp_run_data'))
        tmp_directory_created = True
    
    if download_from_ALPACA.lower() == "missing":
        try:
            downloaded_runs = pl.Series(pl.read_parquet(output_file_path).select("Run Number")).to_list()
        except FileNotFoundError:
            _log.warning(f"{_func_name}:File with the data doesn't exist, trying to download all")
            downloaded_runs = []
    elif download_from_ALPACA.lower() == "all":
        downloaded_runs = []
    else:
        _log.error(f"{_func_name}:Wrong option for donwload_from_ALPACA parameter. Given value: {download_from_ALPACA}. Accepted values are: 'missing','all','none'")
        raise ValueError(f"{_func_name}:Wrong option for donwload_from_ALPACA parameter. Given value: {download_from_ALPACA}. Accepted values are: 'missing','all','none'")
    
    if column_names is None:
        column_names = [name for name in variables_of_interest]
    elif len(variables_of_interest) != len(column_names):
        _log.warning(f"{_func_name}: Number of column names doesn't match number of variables of interest, ignoring the column names")
        column_names = [name for name in variables_of_interest]
        
    # add column with run number
    variables_of_interest = ['Run_Number*Run_Number*__value'] + variables_of_interest
    column_names = ["Run Number"] + column_names
    
    _log.info(f"{_func_name}:Starting downloading data from ALPACA...")
    failed_runs = []
    for run in runs:
        if run in downloaded_runs:
            continue
        if download_from_ALPACA.lower() == "missing":
            try:
                data = pl.read_parquet(os.path.join(data_path,"tmp_run_data",f"tmp_{run}.parquet"))
                _log.info(f"{_func_name}:temporary data file for {run} found at {os.path.join(data_path,'tmp_run_data',f'tmp_{run}.parquet')}")
                continue
            except FileNotFoundError:
                pass            
        data = finalize.generate(first_run=run,
                            last_run= run,
                            elog_results_filename=f'{run}',
                            known_bad_runs=known_bad_runs,
                            verbosing=verbosing,
                            variables_of_interest=variables_of_interest,
                            directories_to_flush=directories_to_flush, 
                            speed_mode=speed_mode)
        try:
            bad_run = False
            for variable in variables_of_interest:
                if isinstance(data[variable.replace('*','_')][0],np.float64) and np.isnan(data[variable.replace('*','_')][0]):
                    data[variable.replace('*','_')][0] = None
                    bad_run = True
            if bad_run:
                failed_runs.append({run:'nan'})
                _log.debug(f"{_func_name}:{run} contains NAN observables, some data might be missing")
            
            data = pl.DataFrame({col_name:data[variable.replace('*','_')] for col_name,variable in zip(column_names,variables_of_interest)})
            _log.debug(data)
            data.write_parquet(os.path.join(data_path,"tmp_run_data",f"tmp_{run}.parquet"))
        except Exception as e:
            failed_runs.append({run:f'{type(e)}'})
            _log.error(f"{_func_name}:{e}")
    
    _log.info(f"{_func_name}:data downloaded")
    if failed_runs:
        _log.warning(f"{_func_name}:Some data might be missing: {failed_runs}")
    
    data:list[pl.DataFrame] = []
    for run in runs:
        if run in downloaded_runs:
            continue
        data.append(pl.read_parquet(os.path.join(data_path,"tmp_run_data",f"tmp_{run}.parquet")))
        for col_name,col_dtype in dtype.items():
            data[-1] = data[-1].with_columns(pl.col(col_name).cast(col_dtype))
    if download_from_ALPACA == "missing" and downloaded_runs:
        data = [pl.read_parquet(output_file_path)] + data
    data:pl.DataFrame = pl.concat(data)
    
    if output_file_name is not None:
        _log.info(f"{_func_name}:Saving data in {output_file_path}")
        data.write_parquet(os.path.join(data_path,"tmp_run_data",output_file_name))
    
    _log.debug(f"{_func_name}:Removing temporary files")
    if cleanup_at_exit:
        for run in runs:
            try:
                os.remove(os.path.join(data_path,"tmp_run_data",f"tmp_{run}.parquet"))
            except FileNotFoundError:
                _log.debug(f"{_func_name}:Run {run} doesn't have a temporary data file")
        try:
            if tmp_directory_created:
                os.rmdir(os.path.join(data_path,'tmp_run_data'))
        except FileNotFoundError:
            _log.debug(f"{_func_name}:temporary directory {os.path.join(data_path,'tmp_run_data')} doesn't exist")
        except OSError:
            _log.debug(f"{_func_name}:temporary directory {os.path.join(data_path,'tmp_run_data')} isn't empty")
    
    return data.filter(pl.col("Run Number").is_in(runs))
    
if __name__ == "__main__":
    ALPACA_load_data()
    
    