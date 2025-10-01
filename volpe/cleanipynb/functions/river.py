import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pandas as pd
import re

def histogramize(data, fld_name, id_run, change = True):

    dt = 2e-9
    t_0 = 0
    dt_bin = 1e-7
    n = int(dt_bin/dt)

    t_i = data[fld_name][id_run]['t']
    v_i = data[fld_name][id_run]['V']
 

    v_data = np.array([np.abs(np.mean(v_i[i*n:(i+1)*n])) for i in range(int(len(v_i)/n))])
    t_data = np.array([t_0 + i*dt_bin for i in range(len(v_data))])

    # print(v_i)
    # return v_i, t_i
    # if change:
    #     return [-1]*v_data+0.01, t_data
    # else:
    return v_data, t_data
    # return [-1]*v_data+0.01, t_data #[-1]*v_data, t_data

def return_m_over_q(voltage = 90):
    _1u2kg_ = 1.66053906660e-27 #kg
    m_over_q= np.arange(1,15) #kg/C
    _C_ = 1.60217662e-19 #C
    U = voltage  #
    d = 1.0 #m
    tof = np.sqrt(m_over_q*_1u2kg_*d**2/(2*_C_*U))
    return tof, m_over_q

# Equation I used: m/q = 2*U*t^2/d^2 --> E_k = 1/2 m v^2 = q U  v = t/d
def TOF_return2(voltage = 90):
    _1u2kg_ = 1.66053906660e-27 #kg
    HCI_mass_u = 39.948#14.0067 #u
    m = HCI_mass_u #* _1u2kg_ kg
    q = np.arange(1,8) #C
    _C_ = 1.60217662e-19 #C
    m_over_q = m/q #kg/C
    U = voltage  #
    d = 1.0 #m
    tof = np.sqrt(m*_1u2kg_*d**2/(2*q*_C_*U))
    return tof, m_over_q

def gauss(x, p): # p[0]==mean, p[1]==stdev
    return 1.0/(p[1]*np.sqrt(2*np.pi))*np.exp(-(x-p[0])**2/(2*p[1]**2))

def mul_gauss(x, a1, m1, s1, a2, m2, s2):
    return a1*np.exp(-(x-m1)**2/(2*s1**2)) + a2*np.exp(-(x-m2)**2/(2*s2**2))

def one_gauss(x, a1, m1, s1):
    return a1*np.exp(-(x-m1)**2/(2*s1**2))

def clean_FFT(ti_bin, vi_bin, dt_bin, idx, cutoff_freq, greater_number):
    '''Czyszczenie FFT'''
    n = len(ti_bin)
    fhat = np.fft.rfft(vi_bin, n)
    fs = 1/dt_bin
    new_freq = np.fft.rfftfreq(n,1/fs)
    L = np.arange(len(new_freq))
    freq = new_freq
    fhat_changed = fhat.copy()
    mask = (freq >= cutoff_freq[idx][0]) & (freq <= cutoff_freq[idx][1]) & (np.abs(fhat) <= greater_number[idx])
    fhat_changed[~mask] = 0
    PSD = np.abs(fhat)
    ffilt = np.fft.irfft(fhat_changed) # Inverse FFT for filtered time signal    
    return ffilt, PSD, freq, fhat_changed, L

  

def where_peaks(ffilt, prom: float, idx: int, exceptions: list = [], exceptions_prom: list = []):

    peaks, _ = find_peaks(ffilt, prominence=prom)# prominence=0.008 height=0.02, prominence=0.01, distance=10)
    # if idx == 1:
    #     peaks, _ = find_peaks(ffilt, prominence=0.003)
    # if idx == 3:
    #     peaks, _ = find_peaks(ffilt, prominence=0.006)
    if exceptions: #exceptions = [3,4]
        if idx in exceptions:
            peaks, _ = find_peaks(ffilt, prominence=exceptions_prom[exceptions.index(idx)])

    print(f'Peaks: {peaks}')
    return peaks

def what_gauss(ti_bin, ffilt, peaks, borders, idx, guess=[0.02, 30e-6, 5e-6]):
    print(f'Peaks: {peaks}, type: {type(peaks)}')
    print(f'Borders: {borders}, type: {type(borders)}')
    print(f'Borders: {borders[idx][0]}, type: {type(borders[idx][0])}')
    start = peaks - borders[idx][0]
    end = peaks + borders[idx][1]
    start = int(start[0])
    end = int(end[0])
    X_fwhm = ti_bin[start:end]
    Y_fwhm = ffilt[start:end]

    i = 1
    while True:
        if ffilt[peaks[0] - i] < 0.0:
            break
        i += 1
    i_start = peaks[0] - i + 1

    i = 1
    while True:
        if ffilt[peaks[0] + i] < 0.0:
            break
        i += 1
    i_end = peaks[0] + i

    X_fwhm_filt = ti_bin[i_start:i_end]
    Y_fwhm_filt = ffilt[i_start:i_end]
    X_fwhm = X_fwhm_filt
    Y_fwhm = Y_fwhm_filt

    guess = [np.max(Y_fwhm), np.mean(X_fwhm), np.std(X_fwhm)]
    p0 = guess
    popt, pcov = curve_fit(one_gauss, X_fwhm, Y_fwhm, p0=p0)

    # Dopasowane parametry
    sigma = np.abs(popt[2])  # Sigma
    FWHM1 = 2.355 * sigma  # FWHM

    # Niepewność sigmy i propagacja błędów
    sigma_err = np.sqrt(np.diag(pcov))[2]  # Niepewność sigmy
    FWHM_err = 2.355 * sigma_err  # Niepewność FWHM

    print(f'mean = {np.abs(popt[1])}, stdev = {sigma} ± {sigma_err}, FWHM = {FWHM1} ± {FWHM_err}')

    y_fit = one_gauss(X_fwhm, *popt)

    return popt, X_fwhm, Y_fwhm, y_fit, FWHM1, FWHM_err

def plot_freq_vs_PSD(freq, PSD, L, ax_psd):
    ax_psd.plot(freq[L], PSD[L], label='noisy')
    ax_psd.set_xlim(freq[L[0]], freq[L[-1]])
    ax_psd.legend()
    ax_psd.set_xlabel('Frequency [Hz]')
    ax_psd.set_ylabel('PSD [V**2/Hz]')

    return ax_psd

def plot_cleaned_data(ti_bin, vi_bin, ffilt, popt, FWHM1, ax, data_number):
    ax.plot(ti_bin, vi_bin, label=f'Original {data_number}')
    ax.plot(ti_bin, ffilt, label=f'Cleaned {data_number}')
    ax.axvspan(popt[1]-FWHM1/2, popt[1]+FWHM1/2, facecolor='g', alpha=0.5)
    # ax.legend()
    ax.set_xlabel(f'TOF {data_number}')
    ax.set_ylabel('Voltage')

    return ax

def plot_clear(ax_clear, ti_bin, ffilt, peaks, data_number):
    ax_clear.plot(ti_bin,ffilt, label=f'Cleaned signal {data_number}')  # Dodajemy każdą serię danych
    # ax_clear.vlines(ti_bin[peaks], -0.010,0)
    ax_clear.legend()
    ax_clear.set_xlabel('TOF')
    ax_clear.set_ylabel('Voltage')
    ax_clear.set_title('Cleaned signals combined')

    return ax_clear

def plot_big(ax_big, ti_bin, vi_bin, ffilt, freq, PSD, L, tof_times, m_ovr_Q, idx, data_number, cutoff_freq, greater_number, dt_bin):

    ax_big[0].plot(ti_bin*1e6, vi_bin, label=f'Original {data_number}')
    ax_big[0].plot(ti_bin*1e6, ffilt, label=f'Cleaned {data_number}')
    ax_big[0].vlines(tof_times*1e6, min(vi_bin),max(vi_bin))
    for i, txt in enumerate(m_ovr_Q):
        ax_big[0].text(tof_times[i]*1e6, 0.6*max(vi_bin), f'm/q = {txt:.2}', rotation=90, size=8)
    #add idx of number above chart
    ax_big[0].title.set_text(f'Run number: {data_number}')
    ax_big[0].legend()
    ax_big[0].set_xlabel('TOF time*1e6 [us]')
    ax_big[0].set_ylabel('Voltage')

    ax_big[1].plot(freq[L], PSD[L], label='noisy')
    ax_big[1].set_xlim(freq[L[0]], freq[L[-1]])
    ax_big[1].set_ylim(min(PSD[L]), max(PSD[L]))
    ax_big[1].hlines(greater_number, freq[L[0]], freq[L[-1]], color='r', linestyle='--')
    ax_big[1].axvline(x=cutoff_freq[0], color='r', linestyle='--', label="Cutoff Frequency")
    ax_big[1].axvline(x=cutoff_freq[1], color='r', linestyle='--')
    ax_big[1].set_xlabel('Frequency (Hz)')
    ax_big[1].set_ylabel('Power Spectral Density')
    ax_big[1].legend()

    ax_big[idx, 2].plot(-PSD[L], freq[L], 'r')  # Odwrócenie osi
    # ax_big[idx, 2].set_title('Odwrócony wykres')
    ax_big[idx, 2].set_xticks([])
    ax_big[idx, 2].set_yticks([])
    ax_big[idx, 2].spines['top'].set_visible(False)
    ax_big[idx, 2].spines['right'].set_visible(False)
    ax_big[idx, 2].spines['bottom'].set_visible(False)
    ax_big[idx, 2].spines['left'].set_visible(False)

    spectrum, freqs, t_spet, _ =  ax_big[3].specgram(vi_bin, NFFT=256, Fs=1/dt_bin, noverlap=128, cmap='jet_r', scale='dB') # scale='dB' 'linear',  'jet_r',cmap='viridis' cmap='inferno'
    ax_big[3].set_xlabel('Time [s]')
    ax_big[3].set_ylabel('Frequency [Hz]')
    ax_big[3].set_label('Spectogram')

    time = ax_big[4].specgram(ffilt, NFFT=256, Fs=1/dt_bin, noverlap=128, cmap='jet_r', scale='dB')
    ax_big[4].set_xlabel('Time [s]')
    ax_big[4].set_ylabel('Frequency [Hz]')
    ax_big[4].set_label('Spectrogram')

    return ax_big #dokonczyc pozniej

def return_peak_setup(peak_setup):
    run = []
    run_values = []
    for i in peak_setup.keys():
        if peak_setup[i] is not None:
            run.append(i)
            run_values.append(peak_setup[i])
    
    return run, run_values


def save_data_to_csv(namefile,columns,args):
    new_data = args
    new_entry = pd.DataFrame([new_data], columns=columns)
    
    try:
        df = pd.read_csv(namefile)
    except FileNotFoundError:
        print("Plik nie istnieje!")

    if new_entry.iloc[0]["Run_number"] in df["Run_number"].values:
        df.loc[df["Run_number"] == new_entry.iloc[0]["Run_number"]] = new_entry.values
    else:
        df = pd.concat([df, new_entry], ignore_index=True)

    # Zapisz całość do pliku
    df.to_csv(namefile, index=False)

    print(f"Dane zapisane poprawnie do pliku {namefile}")

def read_data_from_csv(namefile):
    try:
        df = pd.read_csv(namefile)
    except FileNotFoundError:
        print("Plik nie istnieje!")

    args = df.values
    return args.T


def parse_value(value):
    """Próbuje przekonwertować wartość na listę liczb, jeśli to możliwe."""
    value = value.strip()  # Usuwamy zbędne spacje
    if re.match(r"^\[\s*\d+(\s+\d+)*\s*\]$", value):  # Dopasowanie do formatu listy liczb
        numbers = list(map(int, re.findall(r"\d+", value)))  # Wyciągamy liczby i konwertujemy
        return numbers
    return value  # Jeśli to nie pasuje do listy, zwracamy oryginalną wartość

def read_data_for_run(namefile, run_number):
    try:
        df = pd.read_csv(namefile, dtype=str)  # Wczytujemy jako stringi
    except FileNotFoundError:
        print("Plik nie istnieje!")
        return None
    except Exception as e:
        print(f"Błąd wczytywania pliku: {e}")
        return None

    df = df.applymap(parse_value)  # Parsujemy wartości w całej tabeli
    args = df[df["Run_number"] == str(run_number)].values  # Upewniamy się, że numer runu jest stringiem
    args = args.T
    return args[1:]  # Pomijamy kolumnę z numerem runu   

def return_setup(namefile, run_number):
    args = read_data_for_run(namefile, run_number)
    atom_name = args[0][0]
    borders = args[1][0]
    peak_setup = float(args[2][0])
    cutoff_freq = args[3][0]
    greater_number = float(args[4][0])

    borders = borders.strip("[]")  
    bord = borders.split(",") 
    borders = [float(x.strip()) for x in bord] 
    
    cutoff_freq = cutoff_freq.strip("[]")
    cut = cutoff_freq.split(",")
    cutoff_freq = [float(x.strip()) for x in cut]

    return atom_name, borders, peak_setup, cutoff_freq, greater_number

def set_up(namefile, good_runs):
    type = []
    borders = []
    peak_setup = []
    cutoff_freq = []
    greater_number = []
    for idx in good_runs:
        type_, borders_, peak_setup_,cutoff_freq_, greater_number_ = return_setup('./viktigt/set_up.csv', idx)
        type.append(type_)
        borders.append(borders_)
        peak_setup.append(peak_setup_)
        cutoff_freq.append(cutoff_freq_)
        greater_number.append(greater_number_)
        print(type_, borders_, peak_setup_,cutoff_freq_, greater_number_)

    return type, borders, peak_setup, cutoff_freq, greater_number


def tof_to_energy(TOF): #s

    _1u2kg_ = 1.66053906660e-27 #kg
    HCI_mass_u = 4.002602 #39.948#14.0067 #u  #TODO: log scale and remember about mass
    _J_ = 1.602176634e-19 #J
    m = HCI_mass_u * _1u2kg_ #kg

    # Definiujemy stałe
    # m = 1.66053906660e-27  # kg
    q = 1.602176634e-19  # C
    d = 1.0  # m


    # Obliczamy energię
    # m*v^2/2 = qU -> U = m*d^2/(2*TOF^2*q) -> E = qU -> E = m*d^2/(2*TOF^2) 
    energy = ((m * d**2) / (2 * TOF**2 ))/_J_ #eV

    return energy

def return_m_over_q_He(voltage = 90):
    _1u2kg_ = 1.66053906660e-27 #kg
    HCI_mass_u = 4.002602#14.0067 #u
    m = HCI_mass_u * _1u2kg_ 
    q = 2 #C
    _C_ = 1.60217662e-19 #C
    m_over_q = m/q #kg/C
    U = voltage  #
    d = 1.0 #m
    tof = np.sqrt(m*d**2/(2*q*_C_*U))
    return tof

def energy_to_temperature(energy): 
    # E[eV]=kB[eV/K]*T[K]-> T=E/kB

    k_B = 8.617333262e-5 # eV/K

    # Definiujemy stałe
    # k_B = 1.380649e-23  # J/K
    # q = 1.602176634e-19  # C

    # Obliczamy temperaturę
    temperature = energy / k_B  # K

    return temperature