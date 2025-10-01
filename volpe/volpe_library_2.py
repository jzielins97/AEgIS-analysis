import os
import sys
# from halo import Halo
import pandas as pd
import numpy as np
try:
    #Add path to import ALPACA
    # sys.path.append(os.path.abspath('../../python-analyses'))
    import ALPACA.data.finalize as finalize
except Exception as e:
    print('Error in loading ALPACA')
    # print(e)
from typing import List, Tuple
from IPython.core.getipython import get_ipython
import inspect
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
from scipy.special import erf, erfc
from scipy.stats import exponnorm, norm
import warnings


##################################################################################
#               System functions
##################################################################################

def on_windows():
    if os.name == 'nt':
        return True
    else:
        return False


def create_folder_tree(folder_path):
    # Check if the folder tree exist: if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path) 
        
        
def is_notebook():
    name, is_nb = get_executing_script_or_notebook_name_n_type()
    return is_nb

#     try:
#         # Check if the code is running in a Jupyter notebook
#         shell = get_ipython().__class__.__name__ # type: ignore
#         print(shell)
#         if shell == 'ZMQInteractiveShell':
#             return True  # Jupyter Notebook or JupyterLab
#         elif shell == 'TerminalInteractiveShell':
#             return False  # Running in a normal terminal (python script)
#         else:
#             raise EnvironmentError("Unknown environment")
#     except NameError:
#         # If get_ipython() doesn't exist, it's likely a normal python script
#         return False


def ALPACA_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))) 


# def Databases_path():
#     return os.path.join(ALPACA_path(), 'volpe_analyses', 'Databases')


def get_executing_script_or_notebook_name_n_type() -> Tuple[str, bool]:
    path = None
    is_nb = False
    try:
        # Try getting nb name 
        # NOTE: Only works in VS Code!
        ip = get_ipython()
        if '__vsc_ipynb_file__' in ip.user_ns:          # type: ignore
            path = ip.user_ns['__vsc_ipynb_file__']     # type: ignore
        is_nb = True
    except AttributeError:
        # Get script name
        # Get the current frame
        current_frame = inspect.currentframe()
        # Get the caller's frame (the script that called this function)
        caller_frame = inspect.getouterframes(current_frame)[-1]
        # Get the filename of the caller
        path = caller_frame.filename
        is_nb = False
    except:
        print('Can\'t understand the file name')
        return ('', False) 
    finally:
        name, _ = os.path.splitext(os.path.basename(path)) # type: ignore
        return name, is_nb


def get_executing_script_or_notebook_name():
    name, _ = get_executing_script_or_notebook_name_n_type()
    return name
  
  
##################################################################################
#               ALPACA helpers
##################################################################################
 
def alpaca_run_list(target):
    # Sort the list to ensure it's in order
    target = sorted(target)
    # Get the first and last item
    first_item = target[0]
    last_item = target[-1]
    # Create a full range from the first item to the last
    full_range = set(range(first_item, last_item + 1))
    # Find the missing numbers
    missing_numbers = sorted(full_range - set(target))
    return first_item, last_item, missing_numbers


 
################################################################################## 
#               Ions analyses
##################################################################################

# ToF analyses

def argon_mq(i=0):
    '''
    This function returns m/q for Ag ions up to Ag8+.
    The i term is the Ag(i)+ state. For i = 0, the entire array is returned, where the first item is undefined.
    '''
    Ag_mq = [0,
            39.948,
            19.974,
            13.316,
            9.987,
            7.990,
            6.658,
            5.707,
            4.993]
    if i == 0:
        return Ag_mq
    else:
        return Ag_mq[i]
    

def helium_mq(i=0):
    '''
    This function returns m/q for He ions.
    The i term is the Ag(i)+ state. For i = 0, the entire array is returned, where the first item is undefined.
    '''
    He_mq = [0,4.0026,2.0013]
    if i == 0:
        return He_mq
    else:
        return He_mq[i]    
    
    
def tof_df_creation_from_scratch(runlist: List[int]):
    ''' 
    This function download the data from DAQ with ALPACA and builds the db.
    
    Parametres:
    ------
    runlist: [int] 
        The list of the good runs
        
    Returns:
    --------
    dataframe: pandas.df
        The dataframe with times as indices and in each column the values for each run
    '''
    try:
        first_item, last_item, missing_numbers = alpaca_run_list(runlist)
        # Retrieve the data with ALPACA
        data = finalize.generate(first_run=first_item,
                        last_run=last_item, 
                        elog_results_filename='tof_multifit',
                        known_bad_runs = missing_numbers,
                        verbosing=False,
                        variables_of_interest=[
                            'Run_Number*Run_Number*__value',  # run number
                            'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*t',
                            'captorius1*acq_0*Channel_1_TOF_ions*Y_[V]*V',
                            'Batman*acq_0*Catch_HotStorageTime',
                            'Batman*acq_0*Pbar_CoolingTime'
                            'Batman*acq_0*NestedTrap_SqueezeTime',
                            'Batman*acq_0*NestedTrap_RaiseTime',
                            'Batman*acq_0*NestedTrap_SqueezedTrapType',
                            ],
                            directories_to_flush=['bronze', 'gold', 'datasets', 'elog'],
                            speed_mode=True) #'bronze', 'gold', 'datasets', 'elog'
                            # directories_to_flush=[],
                            # speed_mode=False) 
        # Dataframe creation
        # nb = is_notebook()
        # if not nb: spinner = Halo(text='ToF dataframe creation...', spinner='dots')
        print('Dataframe creation...')
        # if not nb: spinner.start()
        runs = np.array(data['Run_Number_Run_Number___value'])
        # Calculating all the differences of the timepoints: if there's a dicrepancy, a resampling should be done (for now just a warning is shouted)
        diff = len(data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_t'])*data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_t'][0] - np.sum(data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_t'], axis=0)
        if np.any([diff != 0]) > 0:
            print('WARNING! Not all the timebase are identical! Resampling is needed.')
        common_time = data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_t'][0]
        # common_time = np.linspace(data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_t'][0][0],
        #                           data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_t'][0][-1],
        #                           len(data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_t'][0]))
        df = pd.DataFrame({runs[i]:data['captorius1_acq_0_Channel_1_TOF_ions_Y_[V]_V'][i] for i in range(len(runs))}, index=common_time)
        # Transform column name into strings
        df.columns = df.columns.map(str)
        # if not nb: spinner.succeed(f'ToF dataframe created from runs {runs}')
        print('Dataframe created successfully!')
    except Exception as e:
        # if not nb: spinner.fail(e)
        print('Error in creating the dataframe')
        print(e)
        df = pd.DataFrame()
    finally:
        return df
    
        

def tof_df_creation(runlist, db_folder, db_prefix=str(get_executing_script_or_notebook_name())):
    ''' 
    This function creates the dataframe, either loading from the parquet or fetching from ALPACA.
    
    Parametres:
    ------
    runlist: [int] 
        The list of the good runs
        
    Returns:
    ------
    dataframe: pandas.df
        The dataframe with times as indices and in each column the values for each run
    '''
    db_name = db_prefix + '-' + str(runlist[0]) + '-' + str(runlist[-1]) + '.parquet'
    db_path = os.path.join(db_folder, db_name)
    # Check if I've already saved the dataframe
    if os.path.exists(db_path):
        print('Loading the dataframe from the parquet file...')
        try:
            # Load the dataframe and check that the runlist is the same
            df = pd.read_parquet(db_path)
            df_runlist = df.columns.tolist()
            if set(list(map(str,runlist))) == set(df_runlist):
                # Dataframe is good, return it and exit
                print('Dataframe correctly loaded!')
                return df
            else:
                print('Different runs in the runlist, regenerating dataframe')
                print(f'runlist: {runlist}')
                print(f'df runlist: {df_runlist}')
                # continue to dataframe creation
        except Exception as e:
            print('Something went wrong in loading the dataframe:')
            print(e)
            print('Regenerating it.')
            # continue to dataframe creation
    # Create the dataframe
    print('Creating the dataframe from DAQ data...')
    try:
        df = tof_df_creation_from_scratch(runlist)
    except Exception as e:
        print('There was an error in creating the dataframe:')
        print(e)
        # Exiting function
        return pd.DataFrame()
    # Save the dataframe to file
    try:
        create_folder_tree(db_folder)
        df.to_parquet(db_path)
    except Exception as e:
        print('There was an error saving the dataframe:')
        print(e)
        print('Returnig the dataframe anyway')
    finally:
        return df
    
    

def tof_load_df(runlist, db_folder, db_prefix=str(get_executing_script_or_notebook_name()), load_helper=True):
    ''' 
    This function loads the dataframe from the appropriate file.
    
    Parametres:
    ------
    runlist: [int] 
        The list of the good runs
        
    Returns:
    ------
    dataframe: pandas.df
        The dataframe with times as indices and in each column the values for each run
    '''
    # db_folder = Databases_path()
    db_name = db_prefix + '-' + str(runlist[0]) + '-' + str(runlist[-1]) + '.parquet'
    db_helper_name = db_prefix + '-helper-' + str(runlist[0]) + '-' + str(runlist[-1]) + '.parquet'
    db_path = os.path.join(db_folder, db_name)
    db_helper_path = os.path.join(db_folder, db_helper_name)    # Check if I've already saved the dataframe
    if not os.path.exists(db_path):
        print('File not found, generate it from ALPACA.')
        return pd.DataFrame(), pd.DataFrame()
    elif load_helper and not os.path.exists(db_helper_path):
        print('Helper file asked and not found, generate it from ALPACA.')
        return pd.DataFrame(), pd.DataFrame()
    else:        
        print('Loading the dataframe from file...')
        try:
            # Load the dataframe and check that the runlist is the same
            df = pd.read_parquet(db_path)
            df_runlist = df.columns.tolist()
            if load_helper:
                df_helper = pd.read_parquet(db_helper_path)
            else:
                df_helper = pd.DataFrame()
            if set(list(map(str,runlist))) == set(df_runlist):
                # Dataframe is good, return it and exit
                print('Dataframe correctly loaded!')
                return df, df_helper
            else:
                print('Different runs in the runlist, regenerating dataframe')
                print(f'runlist: {runlist}')
                print(f'df runlist: {df_runlist}')
                return pd.DataFrame(), pd.DataFrame()
        except Exception as e:
            print('Something went wrong in loading the dataframe(s):')
            print(e)
            return pd.DataFrame(), pd.DataFrame()
    
    

def mq_from_tof(tof, std_tof = [], V_floor = 180):
    '''
    Function to convert TOF to m/q (in uma/e).
    '''
    t_ref       = 6.493e-6
    std_t_ref   = 0.044e-6
    m_q_ref     = 1.007276
    V_ref       = 150
    mq = ((tof/t_ref)**2) * (V_floor/V_ref) * m_q_ref
    if len(std_tof) == 0:
        return mq
    else:
        std_mq = mq * np.sqrt((2*std_tof/tof)**2 + (2*std_t_ref/t_ref)**2)
        return mq, std_mq


def tof_from_mq(mq,V_floor = 180):
    '''
    Function to convert m/q into tof.
    
    '''
    t_ref       = 6.493e-6
    std_t_ref   = 0.044e-6
    m_q_ref     = 1.007276
    V_ref       = 150
    tof = np.sqrt((mq / m_q_ref) * (V_ref/V_floor)) * t_ref
    return tof
    # mq = ((tof/t_ref)**2) * (V_floor/V_ref) * m_q_ref
    # if len(std_tof) == 0:
    #     return mq
    # else:
    #     std_mq = mq * np.sqrt((2*std_tof/tof)**2 + (2*std_t_ref/t_ref)**2)
    #     return mq, std_mq
    
    
def rebin_df(df, bin_size):
    # # Convert indices to DatetimeIndex
    # df.index = pd.to_datetime(df.index, unit='s')
    # df = df.resample('100ns').sum()
    #######################################################
    # Create bins and group df by them
    n_bins = round((df.index[-1] - df.index[0])/bin_size)
    bins = pd.cut(df.index, bins=n_bins) 
    # grouped = df.groupby(bins, observed=False).mean()  # Average values based on these bins
    # # Set a meaningful index
    # grouped.index = [bin.mid for bin in grouped.index]    
    # # Re-tranform to a dataframe
    # # .to_frame()
    # return grouped
    
    # Group and calculate mean and standard deviation
    grouped = df.groupby(bins, observed=True).agg(['mean', 'std']) #, 'max', 'min'])
    # Flatten the multi-index and assign new names
    grouped.columns = ['values', 'errors'] #, 'max', 'min']
    # Set a meaningful index
    grouped.index = [bin.mid for bin in grouped.index]
    return grouped


def fit_n_plot(df, function, par_i, descriptors):
    '''
    Fit and plot the function on the df
    
    Parametres:
    ------
    function: function(x, par1, par2, ...)
    
    par_i: [float] 
        = [par1_i, par2_i, ...]
        
    descriptors: [string]
    '''
    # Perform the fit
    popt, pcov = curve_fit(function, df.index.values, df.iloc[:,0].values, par_i)
    for descr, value in zip(descriptors, popt):
        print(f'{descr}:\t{value}')
    print(f'Condition number of the covariance matrix: {np.linalg.cond(pcov):e}')
    print('Diagonal values of pcov:')
    for descr, value in zip(descriptors, np.diag(pcov)):
        print(f'Cov of {descr}:\t{value}')
    # Calculate residuals
    residuals = df.iloc[:,0].values - function(df.index.values, *popt)
    # Calculate chi-squared
    chi_squared = np.sum((residuals ** 2) / function(df.index.values, *popt))
    # Calculate R-squared
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((df.iloc[:,0].values - np.mean(df.iloc[:,0].values)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"Chi-squared: {chi_squared}")
    print(f"R-squared: {r_squared}")
    # Generate the fitted curve
    x_fit = np.linspace(min(df.index.values), max(df.index.values), 1000)
    # Plot the data and the fitted curve
    sns.relplot(data=df, label='Data', kind='line', x=df.index, y=df.columns[0])
    plt.plot(x_fit, function(x_fit, *par_i), label='Initial Guess', color='grey')
    plt.plot(x_fit, function(x_fit, *popt), label='Fitted Curve', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    return popt, pcov


def fit_n_plot_errors(df, function, par_i, descriptors, verbose= True, maxfev=5000, xlim = None, ylim = None, yscale = 'linear'):
    '''
    Fit and plot the function on the df
    
    Parametres:
    ------
    df: pandas.Dataframe()
    
        It should have the first column as data and the second column as errors.
        
    function: function(x, par1, par2, ...)
    
        Multiparameteric function for the fit.
    
    par_i: [float] 
        = [par1_i, par2_i, ...]
        
    descriptors: [string]
    '''
    # Perform the fit
    popt, pcov = curve_fit(f = function, 
                           xdata = df.index.values, 
                           ydata = df.iloc[:,0].values, 
                           p0 = par_i, 
                           sigma = df.iloc[:,1].values,
                           absolute_sigma = False,
                           maxfev=maxfev)
    if verbose:
        for descr, value in zip(descriptors, popt):
            print(f'{descr}:\t{value}')
        print(f'Condition number of the covariance matrix: {np.linalg.cond(pcov):e}')
        print('Diagonal values of pcov:')
        for descr, value in zip(descriptors, np.diag(pcov)):
            print(f'Cov of {descr}:\t{value}')
        # Calculate residuals
        residuals = df.iloc[:,0].values - function(df.index.values, *popt)
        # Calculate chi-squared
        # chi_squared = np.sum((residuals ** 2) / function(df.index.values, *popt)) # function from chatgpt, seems wrong
        chi_squared = np.sum((residuals**2)/df.iloc[:,1].values**2)
        dof = len(df)-len(par_i)
        # Calculate R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((df.iloc[:,0].values - np.mean(df.iloc[:,0].values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"Chi-2: {chi_squared}")
        print(f'dof:\t{dof}')
        print(f'Reduced Chi^2:\t{chi_squared/dof}')
        print(f'Goodness-of-Fit:\t{1-chi2.cdf(chi_squared, dof)}')
        print(f"R-squared: {r_squared}")
        # Generate the fitted curve
        x_fit = np.linspace(min(df.index.values), max(df.index.values), 1000)
        # Plot the data and the fitted curve
        plt.figure(figsize=(10, 5))
        plt.errorbar(df.index, df.iloc[:,0].values, yerr=df.iloc[:,1].values, fmt='.', zorder=1)#, markersize=3)
        plt.plot(x_fit, function(x_fit, *par_i), label='Initial Guess', color='grey', zorder=2)
        plt.plot(x_fit, function(x_fit, *popt), label='Fitted Curve', color='red', zorder=3)
        plt.xlabel('x')
        plt.ylabel('y')
        if ylim != None:
            plt.ylim(ylim)
        if xlim != None:
            plt.xlim(xlim)
        if yscale != 'linear':
            plt.yscale(yscale)
        plt.legend()
        plt.show()
    return popt, pcov


def fit_n_plot_errors_for_paper(df, function, par_i, descriptors, verbose= True, maxfev=5000, xlim = None, ylim = None, yscale = 'linear'):
    '''
    Fit and plot the function on the df
    
    Parametres:
    ------
    df: pandas.Dataframe()
    
        It should have the first column as data and the second column as errors.
        
    function: function(x, par1, par2, ...)
    
        Multiparameteric function for the fit.
    
    par_i: [float] 
        = [par1_i, par2_i, ...]
        
    descriptors: [string]
    '''
    # Perform the fit
    popt, pcov = curve_fit(f = function, 
                           xdata = df.index.values, 
                           ydata = df.iloc[:,0].values, 
                           p0 = par_i, 
                           sigma = df.iloc[:,1].values,
                           absolute_sigma = False,
                           maxfev=maxfev)
    if verbose:
        for descr, value in zip(descriptors, popt):
            print(f'{descr}:\t{value}')
        print(f'Condition number of the covariance matrix: {np.linalg.cond(pcov):e}')
        print('Diagonal values of pcov:')
        for descr, value in zip(descriptors, np.diag(pcov)):
            print(f'Cov of {descr}:\t{value}')
        # Calculate residuals
        residuals = df.iloc[:,0].values - function(df.index.values, *popt)
        # Calculate chi-squared
        # chi_squared = np.sum((residuals ** 2) / function(df.index.values, *popt)) # function from chatgpt, seems wrong
        chi_squared = np.sum((residuals**2)/df.iloc[:,1].values**2)
        dof = len(df)-len(par_i)
        # Calculate R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((df.iloc[:,0].values - np.mean(df.iloc[:,0].values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        print(f"Chi-2: {chi_squared}")
        print(f'dof:\t{dof}')
        print(f'Reduced Chi^2:\t{chi_squared/dof}')
        print(f'Goodness-of-Fit:\t{1-chi2.cdf(chi_squared, dof)}')
        print(f"R-squared: {r_squared}")
        # Generate the fitted curve
        x_fit = np.linspace(min(df.index.values), max(df.index.values), 1000)
        # Plot the data and the fitted curve
        plt.figure(figsize=(10, 5))
        plt.errorbar(df.index, df.iloc[:,0].values, yerr=df.iloc[:,1].values, fmt='.', zorder=1)#, markersize=3)
        # plt.plot(x_fit, function(x_fit, *par_i), label='Initial Guess', color='grey', zorder=2)
        plt.plot(x_fit, function(x_fit, *popt), label='Fitted Curve', color='red', zorder=3)
        plt.xlabel('Time of Flight (s)')
        plt.ylabel('Intensity (V)')
        if ylim != None:
            plt.ylim(ylim)
        if xlim != None:
            plt.xlim(xlim)
        if yscale != 'linear':
            plt.yscale(yscale)
        # plt.axvline(x=x0, color='black', linestyle='--')  # Add vertical line at x=x0
        plt.xlim(right=5e-5)  # Cut the plot at x=x0
        plt.xlim(left=0)
        plt.legend()
        plt.show()
    return popt, pcov


def EMG1(x, ampl, mu, sigma, lam):
    return ampl*np.exp((-(x-mu)**2)/(2*sigma**2))*(1+erf(lam*(x-mu)/(np.sqrt(2)*sigma)))


def EMG2(x, ampl, mu, sigma, lam):
    """
    Exponentially Modified Gaussian (EMG) probability density function (PDF).
    
    Parameters:
    - x: points where the PDF is evaluated
    - mu: mean of the Gaussian component
    - sigma: standard deviation of the Gaussian component
    - lam: rate parameter of the exponential component
    
    Returns:
    - PDF values at the given x points
    """
    factor1 = lam / 2
    factor2 = np.exp(factor1 * (2 * mu + lam * sigma**2 - 2 * x))
    factor3 = erfc((mu + lam * sigma**2 - x) / (np.sqrt(2) * sigma))
    total = ampl * factor1 * factor2 * factor3
    # total = np.zeros(len(x))
    # for i, x_i in enumerate(x):
    #     with warnings.catch_warnings(record=True) as w:
    #         warnings.simplefilter("always")
    #         factor1 = lam / 2
    #         factor2 = np.exp(factor1 * (2 * mu + lam * sigma**2 - 2 * x_i))
    #         factor3 = erfc((mu + lam * sigma**2 - x_i) / (np.sqrt(2) * sigma))
    #         total[i] = ampl * factor1 * factor2 * factor3
    #         # Check if any warnings were triggered
    #         if w:
    #             for warning in w:
    #                 if issubclass(warning.category, RuntimeWarning):
    #                     print("Caught RuntimeWarning:", warning.message)
    #                     print(f'for x_i = {x_i}')
    #                     break

    return total


def EMG_exponnorm(x, ampl, loc, scale, K):
    return ampl*exponnorm.pdf(x, K, loc=loc, scale=scale)

def Gauss(x, ampl, loc, scale):
    return ampl*norm.pdf(x, loc=loc, scale=scale)


def print_ToF_details(arr, headers=['Ampl', 'mu', 'sigma', 'lam'], row_headers=None):
    print_in_columns(arr, headers=headers, columns=4, row_headers=row_headers)




##################################################################################
#       Display and printing functions
##################################################################################

def print_in_columns(arr, headers, columns:int, row_headers=None):
    # Ensure the array has a length that is a multiple of columns
    if len(arr) % columns != 0:
        raise ValueError(f'Array length must be a multiple of {columns}.')
    # Ensure headers has exactly column names
    if len(headers) != columns:
        raise ValueError(f'You must provide exactly {columns} headers.')
    # Define a format for printing integers and floats with 3 decimal places
    col_width = 15
    format_str = f"{{:<{col_width}}}"  # Align to left with specified col_width
    float_format = f"{{:<{col_width}.3e}}"  # Align to left with specified col_width and 3 significant digits in scientific notation
    # Print the headers
    if row_headers == None:
        print('-' * col_width * columns)
        print("".join([format_str.format(h) for h in headers]))
    else:
        print('-' * col_width * (columns+1))
        print(' '*col_width + ''.join([format_str.format(h) for h in headers]))  
    # Iterate over the array in chunks of columns
    for i in range(0, len(arr), columns):
        row = arr[i:i+columns]
        formatted_row = []
        # Format each element in the row
        for value in row:
            if isinstance(value, float):
                formatted_row.append(float_format.format(value))
            else:
                formatted_row.append(format_str.format(value))
        # Print the row
        if row_headers==None:
            first_element = ''
        else:
            first_element = format_str.format(row_headers[int(i/columns)])
        print(first_element + ''.join(formatted_row))
    print('-' * col_width * columns)


def print_float_df(df, decimals=3):
    # Define a format for printing floats with specified number of decimal places
    # print(df.map(lambda x: f'{x:.2e}'))
    format_str = f'{{:.{decimals}e}}'
    print(df.map(lambda x: format_str.format(x)))


def massage_df(base_df, bin_size, t0_offset):
    # Sum up the various signals
    df = base_df.mean(axis=1).to_frame()   # NOTE: conversion from Series to DataFrame necessary
    # Remove background signal offset from part before real t0 and reverse the graph
    background = df[df.index < t0_offset].mean()
    df = background - df
    # Resample the dataframe
    df = rebin_df(df, bin_size)
    # Correct the timescale and remove all points before real t0
    df.index -= t0_offset
    df = df[df.index >= 0]
    return df
