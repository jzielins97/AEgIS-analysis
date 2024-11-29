"""Quality of Life Functions

Functions that make life easier, like creating bad runs list from the known good runs. Or chaning the observable name to data key.

"""
from math import sqrt
import numpy as np
from scipy.stats import exponnorm

def to_obsv(observable:str):
    """Return the observable name string for accessing the entries in the finilized data. It changes '*' into '_'

    Args:
        observable (str): input observable string as provided in finalize.generate

    Returns:
        str: observable name
    """
    return observable.replace('*','_')

def make_bad_runs_list(runs):
    """Make array of bad runs based on a list of runs for analysis

    Args:
        runs (_type_): list of good runs
    """
    bad_runs = []
    for i in range(runs[0]+1, runs[-1]):
        if i not in runs:
            bad_runs.append(i)
    
    return bad_runs

def bin_data(bin_size, X, Y, min_X = None, max_X=None)->[np.array,np.array,np.array,np.array]:
    """
    Bin data in X and Y in bin_size bunches. Bin values are calculated as average of all values in the bunch.
    The error in X is calculated as the width of the bin divided by 2.
    The error in Y is calculated as the calculated standard derivation divided by the square root of the number of points in the bin.

    Args:
        bin_size (int): number of points per bunch
        X (np.array): array with x data
        Y (np.array): array with y data
        max_X (float): maximum x value for any bin. This cuts the spectrum. Leaving it to None, the full spectrum is binned.

    Returns:
        Tuple[X_binned,Y_binned,errX,errY]: binned X and Y data, plus errors in X and Y
    """
    X_binned = []
    Y_binned = []
    errY = []
    # iterate over the data in steps of the bin_size
    for i in range(len(X)//bin_size):
        # check limits
        if min_X is not None:
            if X[i*bin_size] < min_X:
                continue
        if max_X is not None:
            if X[i*bin_size] > max_X:
                break # the x_top value is higher than the limit

        # calculate means
        X_binned.append(np.mean(X[i*bin_size:(i+1)*bin_size]))
        Y_binned.append(np.mean(Y[i*bin_size:(i+1)*bin_size]))
        errY.append(np.std(Y[i*bin_size:(i+1)*bin_size]))

    leftovers = len(X)%bin_size
    # append leftovers as another point
    if leftovers > 0:
        # calculate means
        X_binned.append(np.mean(X[(i+1)*bin_size:]))
        Y_binned.append(np.mean(Y[(i+1)*bin_size:]))
        errY.append(np.std(Y[(i+1)*bin_size:]))

    # calculate stds for errors
    errX = np.full(len(X_binned),(X[bin_size]-X[0])/2)
    # errY = np.array([y/sqrt(bin_size) for y in errY])
    errY = np.array([y/sqrt(bin_size) for y in errY])
    return np.array(X_binned), np.array(Y_binned), errX, errY
    
def spectrum_fit(x,*param):
    return param[0]*(exponnorm.pdf(x,param[1],param[2],param[3])+param[4]) # + param[0+4]*exponnorm.pdf(x,param[1+4],param[2+4],param[3+4])

def linear(x,*param):
    return param[0] + param[1]*x

def default_figure(plt):
    SMALL_SIZE = 25
    MEDIUM_SIZE = 30
    BIGGER_SIZE = 50
    # plt.figure(dpi=1200)
    #sns.set_context('talk')

    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title
    plt.rc('grid',linewidth=0.5)
    plt.rc('grid',color="0.5")
    plt.rc('lines',linewidth=1)
    plt.rc('savefig',dpi=1200)