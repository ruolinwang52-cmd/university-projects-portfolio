# Imports
import numpy as np
import scipy.signal
import scipy.io as sio
from scipy.fft import fft, ifft
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
# Data loading for the exercise
path_to_data = "sunspotdata.mat" # Change to YOUR path to data.
mat = sio.loadmat(path_to_data, simplify_cells=True)
dataold = mat["dataold"]
datanew = mat["datanew"]
timeold = mat["timeold"]
timenew = mat["timenew"]

path_to_data = "eegsingle.mat" # Change to YOUR path to data.
mat = sio.loadmat(path_to_data, simplify_cells=True)
eegsingle = mat["eeg"]
time = mat["time"]

path_to_data = "eegmulti.mat" # Change to YOUR path to data.
mat = sio.loadmat(path_to_data, simplify_cells=True)
eegmulti = mat["eegmat"]
channels = mat["channels"]

# 1.1 Investigation of spectral estimation techniques for simulated data



e = np.random.randn(1000, 1)
A = [1, -2.39, 3.35, -2.34, 0.96]
C = [1, 0, 0.999]
x = scipy.signal.lfilter(C, A, e, axis=0)
x = x[500:].flatten()
plt.plot(x)
plt.title('500-sample realization of ARMA(4,2) process')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()

# Sampling frequency
fs = 100

# Create a time vector for the 500 samples
tvect = np.arange(0, len(x)) / fs

# Plot the ARMA data with the corresponding time vector
plt.plot(tvect, x)
plt.title('ARMA(4,2) process vs Time')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()

# Zero-padding to FFT length 4096
nfft = 4096

# Compute the periodogram (with zero-padding to nfft)
frequencies, power_spectral_density = scipy.signal.periodogram(x, fs=fs, nfft=nfft)
psd_dB = 10 * np.log10(power_spectral_density)
plt.plot(frequencies, psd_dB)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.title('Periodogram-based Spectral Density (ARMA(4,2))')
plt.grid(True)
plt.ylim([-100, 50])
plt.show()






w, h = scipy.signal.freqz(C, A, worN=4096)
# Calculate the power spectral density from the frequency response
R = np.abs(h)**2
# Plot the theoretical spectral density (true spectrum) in dB scale
plt.plot(w / (2 * np.pi) * fs, 10 * np.log10(R / fs))
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density (dB)")
plt.ylim([-100, 50])  # Adjust y-axis limits
plt.title("The true spectral density of the ARMA(4,2) process")
plt.show()



# Generate a new noise sequence and compare 
e_new = np.random.randn(1000, 1)
x_new = scipy.signal.lfilter(C, A, e_new, axis=0)
x_new = x_new[500:].flatten()
frequencies_new, power_spectral_density_new = scipy.signal.periodogram(x_new, fs=fs, nfft=nfft)
psd_dB_new = 10 * np.log10(power_spectral_density_new)
plt.figure(figsize=(10, 6))
plt.plot(frequencies, psd_dB, label='Periodogram 1', color='b')
plt.plot(frequencies_new, psd_dB_new, label='Periodogram with noise', color='g')
plt.plot(w / (2 * np.pi) * fs, 10 * np.log10(R / fs), label='True PSD', color='black')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.title('Comparison of Periodograms and True Spectral Density')
plt.legend()
plt.grid(True)
plt.ylim([-100, 50])


#Q1 The bias is smaller between about 1Hz to 18Hz, and the bias is significantly large otver 20Hz. 
#The periodogram is known to have high variance, 
#particularly when using a single realization



# Compute the periodogram using a Hanning window
frequencies_hann, psd_hann = scipy.signal.periodogram(x_new, fs=fs, nfft=4096, window='hann', axis=0)
psd_hann_dB = 10 * np.log10(psd_hann)
plt.plot(frequencies_hann, psd_hann_dB, label='Hanning-windowed Periodogram', color='purple')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.title('Comparison of Hanning-windowed Periodogram and True Spectral Density')
plt.grid(True)
plt.legend()
plt.ylim([-100, 50])
plt.show()


#Q2 Hanning window has lower sidelobes, but a slightly wider mainlobe. 
#Sidelobes represent the amount of leakage from one frequency to another.


#1.2 The balance between resolution and variance reduction of periodogram-based methods
# Q3 
#N = (L - \text{overlap}) \times \text{number of windows}
#500= N = (L - \frac{L}{2}) \times 10 = \frac{L}{2} \times 10
#L=100


# Given parameters
fs = 100  
L = 100  # Window length
nfft = 4096  # FFT length
e_new = np.random.randn(1000, 1)
x_new = scipy.signal.lfilter(C, A, e_new, axis=0)
x_new = x_new[500:].flatten()
# Apply Welch method with Hanning window and 50% overlap
f, Pxx = scipy.signal.welch(x_new, fs=fs, nfft=nfft, window='hann', nperseg=L, noverlap=L//2, axis=0)
Pxx_dB = 10 * np.log10(Pxx)
plt.plot(f, Pxx_dB, label='Welch Periodogram', color='purple')
# Plot the true spectral density for comparison
w, h = scipy.signal.freqz(C, A, worN=4096)
R = np.abs(h)**2
plt.plot(w / (2 * np.pi) * fs, 10 * np.log10(R / fs), label='True PSD', color='r')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.title('Comparison of Welch Periodogram and True Spectral Density')
plt.grid(True)
plt.legend()
plt.ylim([-100, 50])
plt.show()

#Q4 As we see that the variance of periodogram is significantly reduceds.
#Q5 Welch periodogram cannot catch sharp peaks, it will smooth out. In the true PSD, there are distinct peaks, 
#while the Welch periodogram (purple line) smooths out these peaks.
#A shorter window (like L = 100 here) 
#improves variance reduction but bad in terms of frequency resolution. A shorter window leads to a wider mainlobe. 


#1.3 Comparison of variances for different methods
# Sampling frequency and parameters
fs = 100
nfft = 4096
L = 100  # Window length for Welch method
e = np.random.randn(1000, 1).flatten()
# Compute the periodogram using a Hanning window (modified periodogram)
f_period, P_period = scipy.signal.periodogram(e, fs=fs, nfft=nfft, window='hann')
# Compute the Welch method periodogram with Hanning window and 50% overlap
f_welch, P_welch = scipy.signal.welch(e, fs=fs, nfft=nfft, window='hann', nperseg=L, noverlap=L//2)
var_period = np.var(P_period)
var_welch = np.var(P_welch)
# Calculate variance ratio
variance_ratio = var_period / var_welch
print(f"Variance of Periodogram: {var_period}")
print(f"Variance of Welch Estimate: {var_welch}")
print(f"Variance Ratio (Periodogram/Welch): {variance_ratio}")
# Plot the periodogram and Welch estimates for comparison
plt.figure(figsize=(10, 6))
plt.plot(f_period, 10 * np.log10(P_period), label='Modified Periodogram', color='blue')
plt.plot(f_welch, 10 * np.log10(P_welch), label='Welch Method', color='purple')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (dB)')
plt.title('Comparison of Modified Periodogram and Welch Method')
plt.legend()
plt.grid(True)
plt.show()


#Q6 Welch method gives a much smoother spectral estimate. 
#The modified periodogram uses the entire data sequence without averaging, leading to a higher variance





#2.1 Sunplot data and the advantage of zero-mean data.
mat = sio.loadmat('sunspotdata.mat', simplify_cells=True)
dataold = mat["dataold"]
datanew = mat["datanew"]
timeold = mat["timeold"]
timenew = mat["timenew"]
plt.figure(figsize=(10, 6))
plt.plot(timeold, dataold, label='Data Old (including 1859)')
plt.title('Sunspot Data Old Sequence')
plt.xlabel('Years')
plt.ylabel('Sunspot Counts')
plt.grid(True)
plt.legend()
plt.figure(figsize=(10, 6))
plt.plot(timenew, datanew, label='Data New (Recent Years)')
plt.title('Sunspot Data New Sequence')
plt.xlabel('Years')
plt.ylabel('Sunspot Counts')
plt.grid(True)
plt.legend()
plt.show()
# Find peaks for dataold
peaks_old, _ = find_peaks(dataold, distance=12*8)  # Use minimum distance between peaks of about 8 years 
peak_times_old = timeold[peaks_old]  # Extract time corresponding to peaks
estimated_period_old_years = np.mean(np.diff(peak_times_old))  # Calculate average period in years
# Find peaks for datanew
peaks_new, _ = find_peaks(datanew, distance=12*8)  # Use minimum distance between peaks of about 8 years 
peak_times_new = timenew[peaks_new]  # Extract time corresponding to peaks
estimated_period_new_years = np.mean(np.diff(peak_times_new))  # Calculate average period in years
# Convert periods from years to months
estimated_period_old_months = estimated_period_old_years * 12
estimated_period_new_months = estimated_period_new_years * 12
print(f"Estimated oscillation period for dataold: {estimated_period_old_months:.2f} months (~{estimated_period_old_years:.2f} years)")
print(f"Estimated oscillation period for datanew: {estimated_period_new_months:.2f} months (~{estimated_period_new_years:.2f} years)")
# Estimate the corresponding frequencies (1/period) in cycles per month
frequency_old = 1 / estimated_period_old_months
frequency_new = 1 / estimated_period_new_months
print(f"Estimated frequency for dataold: {frequency_old:.4f} cycles per month")
print(f"Estimated frequency for datanew: {frequency_new:.4f} cycles per month")

#Q7 Answers in Console, we see the peaks comes more often in datanew, but cycles are not that accurate.


def plot_periodogram(data, label):
    # Compute FFT
    X = fft(data)
    N = len(data)
    # Compute periodogram (power spectral density)
    Rhat = (X * np.conj(X)) / N  # Spectral density
    f = np.arange(N) / N  # Normalized frequency axis
    plt.figure(figsize=(10, 6))
    plt.plot(f[:N//2], np.real(Rhat[:N//2]), label=label)  # Only plot the positive frequencies
    plt.title(f"Periodogram of {label}")
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Power Spectral Density')
    plt.grid(True)
    plt.legend()
    plt.show()
plot_periodogram(dataold, 'Data Old (including 1859)')
plot_periodogram(datanew, 'Data New (Recent Years)')
def plot_periodogram_zoom(data, label, freq_range=(0, 0.05)):
    # Compute FFT
    X = fft(data)
    N = len(data)
    Rhat = (X * np.conj(X)) / N  # Spectral density
    f = np.arange(N) / N  # Normalized frequency axis
    # Plot periodogram
    plt.figure(figsize=(10, 6))
    plt.plot(f[:N//2], np.real(Rhat[:N//2]), label=label)  # Only plot positive frequencies
    plt.xlim(freq_range)  # Zoom into the frequency range of interest
    plt.title(f"Periodogram of {label} (Zoomed)")
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Power Spectral Density')
    plt.grid(True)
    plt.legend()
    plt.show()
plot_periodogram_zoom(dataold, 'Data Old (including 1859)', freq_range=(0.007, 0.015))
plot_periodogram_zoom(datanew, 'Data New (Recent Years)', freq_range=(0.001, 0.015))

#Q8 	The very high peak at or near zero frequency represents the mean of data is non-zero. We can roughly see
# the estimated frequency in the zoomed graph but not very clear.


# Remove the mean (zero-mean correction)
data0_old = dataold - np.mean(dataold)
data0_new = datanew - np.mean(datanew)
def plot_periodogram_zoom(data, label, freq_range=(0, 0.05)):
    # Compute FFT
    X = fft(data)
    N = len(data)
    
    # Compute periodogram (power spectral density)
    Rhat = (X * np.conj(X)) / N  # Spectral density
    f = np.arange(N) / N  # Normalized frequency axis
    plt.figure(figsize=(10, 6))
    plt.plot(f[:N//2], np.real(Rhat[:N//2]), label=label)  # Only plot positive frequencies
    plt.xlim(freq_range)  # Zoom into the frequency range of interest
    plt.title(f"Periodogram of {label} (Zero-Mean Corrected)")
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Power Spectral Density')
    plt.grid(True)
    plt.legend()
    plt.show()
plot_periodogram_zoom(data0_old, 'Zero-Mean Data Old (including 1859)', freq_range=(0.007, 0.015))
plot_periodogram_zoom(data0_new, 'Zero-Mean Data New (Recent Years)', freq_range=(0.000, 0.08))



#Q9 Now we can see more clear about the peak frequency.

#2.2 The advantage of zero-padding data

 #Function to compute and plot the periodogram with zero-padding and zoom
def plot_periodogram_zeropad(data, label, nfft=8192, freq_range=(0, 0.05)):
    # Compute FFT with zero-padding
    X = fft(data, n=nfft)
    N = len(data)
    Rhat = (X * np.conj(X)) / N  # Normalize by original data length N
    f = np.arange(nfft) / nfft  # Normalized frequency axis for nfft points
    plt.figure(figsize=(10, 6))
    plt.plot(f[:nfft//2], np.real(Rhat[:nfft//2]), label=label)  # Only plot positive frequencies
    plt.xlim(freq_range)  # Zoom into the frequency range of interest
    plt.title(f"Zero-Padded Periodogram of {label} (nfft={nfft})")
    plt.xlabel('Normalized Frequency')
    plt.ylabel('Power Spectral Density')
    plt.grid(True)
    plt.legend()
    plt.show()
plot_periodogram_zeropad(data0_old, 'Zero-Mean Data Old (including 1859)', nfft=8192, freq_range=(0.007, 0.015))
plot_periodogram_zeropad(data0_new, 'Zero-Mean Data New (Recent Years)', nfft=8192, freq_range=(0.007, 0.015))

#Q10 The strongest peaks in both periodograms appear around a normalized frequency close to 0.008 and 0.0073,
#which is in line with the general range of the estimated periods in Q7. Zero-padding is better for shorter signals.


