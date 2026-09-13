import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import scipy.signal
import sounddevice as sd
from scipy.signal import periodogram
from scipy.signal import spectrogram
from help_funcs_SSP import signalsim, getCov, melSpectrogram, melFil
plt.close('all')

#1.1
path_to_data = "data.mat" 
data = scipy.io.loadmat(path_to_data,simplify_cells=True)
print(data.keys())

plt.figure(figsize=(8, 4))
plt.plot(data["data1"], label='Realization 1')
plt.plot(data["data2"], label='Realization 2')
plt.plot(data["data3"], label='Realization 3')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.title('Realizations of White Gaussian Noise')
plt.legend()
plt.show()

#Q1 From graph it seems like from the same realization, does not have 0 mean


m1 = np.mean(data["data1"],axis=0)
m2 = np.mean(data["data2"],axis=0)
m3 = np.mean(data["data3"],axis=0)
print('The mean value for realization 1 is')
print(m1)
print('The mean value for realization 2 is')
print(m2)
print('The mean value for realization 3 is')
print(m3)


# 已知的方差和样本大小
variance = 0.25
std_dev = np.sqrt(variance)
n = len(data["data1"])  # 假设所有数据集的长度相同

# 95%置信水平的临界值
z = 1.96

# 标准误差
SE = std_dev / np.sqrt(n)

# 置信区间计算
# 95%置信区间
CI1 = (m1 - z * SE, m1 + z * SE)
CI2 = (m2 - z * SE, m2 + z * SE)
CI3 = (m3 - z * SE, m3 + z * SE)

# Use formatted strings to control decimal places
print(f"mean 1 95% confidence intervals: ({CI1[0]:.3f}, {CI1[1]:.3f})")
print(f"mean 2 95% confidence intervals: ({CI2[0]:.3f}, {CI2[1]:.3f})")
print(f"mean 3 95% confidence intervals: ({CI3[0]:.3f}, {CI3[1]:.3f})")

# Q2 0 is outside the confidence interval, so none of process has underlying 0 mean
overall_mean = np.mean([m1,m2,m3]) #average of mean of all realizations
print('Average of mean of all realizations')
print(overall_mean) 
#Q3


#1.2
# Input code for loading 'data2.mat'
path_to_data = "data2.mat" # Change to YOUR path to data.
data = scipy.io.loadmat(path_to_data,simplify_cells=True)


max_lag = 10
r, lags = getCov(data["y"],max_lag,"r")
plt.figure(figsize=(8, 4))
plt.plot(lags,r, label='Auto-covariance')
plt.xlabel('lags')
plt.ylabel('Covariance')
plt.title('Covariance function')
plt.legend()
plt.show()

rho, lags = getCov(data["y"],max_lag,"rho")
plt.figure(figsize=(8, 4))
plt.plot(lags,rho, label='Correlation')
plt.xlabel('lags')
plt.ylabel('Correlation')
plt.title('Correlation function')
plt.legend()
plt.show()
plt.close()


#Q4 correlation fcn = covariance / 0.25, which is normalaized 


k = 1 # sets the lag
plt.figure(figsize=(8, 4))
plt.scatter(data["y"][:-k],data["y"][k:], label='Realization 1')
plt.show()

k = 2 # sets the lag
plt.figure(figsize=(8, 4))
plt.scatter(data["y"][:-k],data["y"][k:], label='Realization 1')
plt.show()

k = 3 # sets the lag
plt.figure(figsize=(8, 4))
plt.scatter(data["y"][:-k],data["y"][k:], label='Realization 1')
plt.show()
plt.close()


#Q5 As we see second scatter plot, in the beginning data are scattered so in the beginning of cor fcn it is close to 0

# 设定参数
fs = 256  # 采样频率
N = 500   # 样本数量
t = np.arange(N) / fs  # 时间数组

# 定义频率和随机参数
f1 = 10  # 频率1
f2 = 20  # 频率2

# 随机相位
phi1 = np.random.uniform(0, 2 * np.pi)
phi2 = np.random.uniform(0, 2 * np.pi)

# 独立振幅
A1 = np.random.rayleigh(scale=2)
A2 = np.random.rayleigh(scale=2)

# 生成信号
#x = A1 * np.cos(2 * np.pi * f1 * t + phi1) + A2 * np.cos(2 * np.pi * f2 * t + phi2)

# 可视化信号
#plt.plot(t, x)
#plt.xlabel('Time (s)')
#plt.title('Simulated Realization')
#plt.show()



# Use following to lines...: 
x, t = signalsim()
# ... Or create your own method of simulating the process above:

# Visualize the realization
plt.plot(t, x)
plt.xlabel('Time (s)')
plt.title('Realization')
plt.show()

#Q6 Maximum time value in seconds = N/f_s

# 假设 x 和 t 是之前代码中生成的信号和时间数组
# 定义最大滞后时间（秒），如 1 秒
max_lag_seconds = 1
max_lag_samples = int(max_lag_seconds * fs)  # 最大滞后时间转换为样本数

# 计算最大滞后的协方差
def covariance(x, max_lag_samples):
    N = len(x)
    mean_x = np.mean(x)
    covariances = []
    for k in range(max_lag_samples):
        cov = np.sum((x[:N-k] - mean_x) * (x[k:] - mean_x)) / (N - k)
        covariances.append(cov)
    return np.array(covariances)

cov_values = covariance(x, max_lag_samples)

# 将滞后样本数转换为时间（秒）
lags_in_seconds = np.arange(max_lag_samples) / fs

# Plot the covariance function
plt.figure()
plt.plot(lags_in_seconds, cov_values)
plt.xlabel('Lag (seconds)')
plt.ylabel('Covariance')
plt.title('Covariance Function')
plt.show()

fs = 256
nfft = 2048
f,P = scipy.signal.periodogram(x,fs=fs,nfft=nfft)
plt.plot(f,P)
plt.xlabel('frequency [Hz]')
plt.ylabel('Periodogram')
plt.show()

plt.plot(f,10*np.log10(P))
plt.xlabel('frequency [Hz]')
plt.ylabel('10*log10(Power)')
plt.ylim([-60,np.max(10*np.log10(P))*1.1])
plt.show()





#Q7 yes, the period is about 0.1












path_to_data = "cellodiffA.mat" 
data = scipy.io.loadmat(path_to_data,simplify_cells=True)


# Extract the cello notes
celloA2 = data['celloA2'].squeeze()
celloA3 = data['celloA3'].squeeze()
celloA4 = data['celloA4'].squeeze()

# Sampling frequency
fs = 44100  # Hz

# Function to plot periodogram in dB scale
def plot_periodogram(signal, fs, title):
    f, Pxx = periodogram(signal, fs=fs)
    
    # Replace zero or negative values in Pxx with a small positive value to avoid log errors
    Pxx[Pxx <= 0] = 1e-10
    
    Pxx_dB = 10 * np.log10(Pxx)
    plt.figure(figsize=(8, 4))
    plt.plot(f, Pxx_dB)  # Changed from semilogy to plot to avoid log scaling issues
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power/Frequency [dB/Hz]')
    plt.grid(True)
    plt.show()


# Play the sound for verification (optional)
#sd.play(celloA2, fs)
#sd.play(celloA3, fs)
#sd.play(celloA4, fs)

# Plot periodograms for each note
plot_periodogram(celloA2, fs, 'Spectral Density - celloA2')
plot_periodogram(celloA3, fs, 'Spectral Density - celloA3')
plot_periodogram(celloA4, fs, 'Spectral Density - celloA4')


#Q11  The overtones are the peaks that appear at integer multiples of the keynote frequency. 	•	
   # For celloA2 (110 Hz):Peaks after 110 Hz should appear at approximately 220 Hz, 330 Hz, 440 Hz and so on.
	#For celloA3 (220 Hz):Overtones should appear at approximately 440 Hz, 660 Hz, and higher 
	#For celloA4 (440 Hz):Overtones should appear at 880 Hz, 1320 Hz, and so on.






# Load the data
path_to_data = 'celloandflute.mat'  
data = scipy.io.loadmat(path_to_data)

# Extract the signals for each instrument 
cello1 = data['celloA41'].squeeze()
cello2 = data['celloA42'].squeeze()
flute1 = data['fluteA41'].squeeze()
flute2 = data['fluteA42'].squeeze()

# Sampling frequency
fs = 44100  # Hz

# Function to plot the periodogram in dB scale
def plot_periodogram(signal, fs, title):
    f, Pxx = periodogram(signal, fs=fs)
    
    # Replace zero or negative values in Pxx with a small positive value to avoid log errors
    Pxx[Pxx <= 0] = 1e-10
    
    Pxx_dB = 10 * np.log10(Pxx)
    plt.figure(figsize=(8, 4))
    plt.plot(f, Pxx_dB)
    plt.title(title)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Power/Frequency [dB/Hz]')
    plt.grid(True)
    plt.show()

# Plot periodograms for all four cases
plot_periodogram(cello1, fs, 'Spectral Density - cello1')
plot_periodogram(cello2, fs, 'Spectral Density - cello2')
plot_periodogram(flute1, fs, 'Spectral Density - flute1')
plot_periodogram(flute2, fs, 'Spectral Density - flute2')

#Q12 Cello has more overtones than flute which gives it richer harmonic tones. One can tell the difference by looking at numbers of harmonic overtones. 




path_to_data = 'cellomelody.mat' 
data = scipy.io.loadmat(path_to_data)


melody1 = data['melody1'].squeeze()
melody2 = data['melody2'].squeeze()
fs = data['fs'].squeeze()  # Sampling frequency

#sd.play(melody1, fs)
#sd.play(melody2, fs)

# Set parameters
window = 2048  # Window length (sequence length of a periodogram)
nfft = 16384  # Number of frequency bins for the FFT
noverlap = 1024  # Number of overlapping samples

# Compute and plot the spectrogram for melody1
f, t, S1 = scipy.signal.spectrogram(melody1, fs=fs, window='hann', nperseg=window, noverlap=noverlap, nfft=nfft)
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, 10 * np.log10(S1), shading='gouraud')
plt.title('Spectrogram - Melody 1')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim([0, 2000])  # Zoom into frequencies below 2 kHz to focus on keynotes
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.show()

# Compute and plot the spectrogram for melody2
f, t, S2 = scipy.signal.spectrogram(melody2, fs=fs, window='hann', nperseg=window, noverlap=noverlap, nfft=nfft)
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f, 10 * np.log10(S2), shading='gouraud')
plt.title('Spectrogram - Melody 2')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.ylim([0, 2000])  # Zoom into frequencies below 2 kHz to focus on keynotes
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.show()

#Q13    Distribution of energy across frequencies are different. If one melody is rhythmically different, you will see distinct patterns of time gaps or more frequent changes in the spectrogram




# Function to plot spectrograms with different window and overlap settings
def plot_spectrogram(melody, window, noverlap, nfft, title):
    f, t, S = scipy.signal.spectrogram(melody, fs=fs, window='hann', nperseg=window, noverlap=noverlap, nfft=nfft)
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10 * np.log10(S), shading='gouraud')
    plt.title(title)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.ylim([0, 2000])  # Zoom into frequencies below 2 kHz
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.show()


#Q14  Long window (better frequency resolution, worse time resolution);Short window (better time resolution, worse frequency resolution)
# Parameters for different window lengths
nfft = 16384  # Same NFFT for both cases

long_window = 4096  # Large window
long_noverlap = 2048  # 50% overlap
plot_spectrogram(melody1, long_window, long_noverlap, nfft, 'Spectrogram - Melody 1 (Long Window)')
plot_spectrogram(melody2, long_window, long_noverlap, nfft, 'Spectrogram - Melody 2 (Long Window)')


short_window = 512  # Small window
short_noverlap = 256  # 50% overlap
plot_spectrogram(melody1, short_window, short_noverlap, nfft, 'Spectrogram - Melody 1 (Short Window)')
plot_spectrogram(melody2, short_window, short_noverlap, nfft, 'Spectrogram - Melody 2 (Short Window)')






path_to_data = 'cellomelody.mat' 
data = scipy.io.loadmat(path_to_data)

# Extract the melodies and the sampling frequency
melody1 = data['melody1'].squeeze()
melody2 = data['melody2'].squeeze()
fs = data['fs'].squeeze()  # Sampling frequency

# Set the parameters for the mel filters
nfft = 16384  # NFFT used for the spectrogram
noverlap = 1024  # Overlap for the spectrogram

# Define nperseg larger than noverlap
nperseg = 2048  # Window length, must be greater than noverlap

# 32 mel filters for reduced frequency resolution
mel_filters = melFil(fs, nfft)

# Step 1: Compute the spectrogram for melody1 and melody2
f1, t1, Sxx1 = spectrogram(melody1, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window='hann')
f2, t2, Sxx2 = spectrogram(melody2, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, window='hann')

# Step 2: Apply the mel filters to the spectrogram
melS1 = np.dot(mel_filters, Sxx1)
melS2 = np.dot(mel_filters, Sxx2)

# Step 3: Plot the mel spectrogram for melody1
plt.figure(figsize=(10, 6))
plt.imshow(10 * np.log10(melS1), aspect='auto', origin='lower')
plt.title('Mel Spectrogram - Melody 1')
plt.ylabel('Mel Frequency')
plt.xlabel('Time [sec]')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.show()

# Step 3: Plot the mel spectrogram for melody2
plt.figure(figsize=(10, 6))
plt.imshow(10 * np.log10(melS2), aspect='auto', origin='lower')
plt.title('Mel Spectrogram - Melody 2')
plt.ylabel('Mel Frequency')
plt.xlabel('Time [sec]')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.show()

#Q15 Really bad resolution

#Melody 1 has more frequent transitions and appears more active or varied over time in the lower frequencies.
#Melody 2 shows longer periods of sustained notes, with less frequent transitions, visible as longer horizontal bands.

#The frequency resolution in the higher ranges is reduced, which could lead to difficulties in distinguishing instruments or melodies with rich harmonic content such as cello.
# difficult to differentiating two instruments with very similar timbres but different overtones





path_to_data = 'cellomelody.mat' 
data = scipy.io.loadmat(path_to_data)

# Extract the melodies and the sampling frequency
melody1 = data['melody1'].squeeze()
fs = data['fs'].squeeze()  # Sampling frequency

# 1) Decimation
q = 8  # Decimation factor
melody1_decimated = scipy.signal.decimate(melody1, q)

# 2) Playback (Make sure you adjust the sampling rate after decimation)
fs_decimated = fs // q  # Adjusted sampling frequency after decimation
#sd.play(melody1_decimated, fs_decimated)  # Play the decimated melody
#
#sd.wait()  # Wait until playback is finished

# 3) Spectrogram of Original Data
f1, t1, Sxx1 = scipy.signal.spectrogram(melody1, fs=fs, nperseg=512, noverlap=256)
plt.figure(figsize=(10, 6))
plt.pcolormesh(t1, f1, 10 * np.log10(Sxx1), shading='gouraud')
plt.title('Spectrogram - Original Melody 1')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.show()

# 4) Spectrogram of Decimated Data
f2, t2, Sxx2 = scipy.signal.spectrogram(melody1_decimated, fs=fs_decimated, nperseg=512, noverlap=256)
plt.figure(figsize=(10, 6))
plt.pcolormesh(t2, f2, 10 * np.log10(Sxx2), shading='gouraud')
plt.title('Spectrogram - Decimated Melody 1 (q=8)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.show()

#Q16  Yes, the keynote (fundamental) frequencies are the same in both images, as they are located in the same frequency range in both spectrograms. However, the higher harmonics (overtones) have been lost in the decimated version


path_to_data = 'cellomelody.mat' 
data = scipy.io.loadmat(path_to_data)

# Extract the melodies and the sampling frequency
melody1 = data['melody1'].squeeze()
fs = data['fs'].squeeze()  # Sampling frequency

# Decimation with low-pass filter (using scipy.signal.decimate)
q = 8  # Decimation factor
melody1dec = scipy.signal.decimate(melody1, q)

# Decimation without low-pass filtering (direct downsampling)
melody1alias = melody1[::q]

# Adjust the new sampling rate after decimation
fs_decimated = fs // q

# Playback decimated melodies 
#sd.play(melody1dec, fs_decimated)  
# sd.wait()
#sd.play(melody1alias, fs_decimated) 
# sd.wait()

# Spectrogram of Melody 1 Decimated (with low-pass filter)
f1, t1, Sxx1 = scipy.signal.spectrogram(melody1dec, fs=fs_decimated, nperseg=512, noverlap=256)
plt.figure(figsize=(10, 6))
plt.pcolormesh(t1, f1, 10 * np.log10(Sxx1), shading='gouraud')
plt.title('Spectrogram - Decimated Melody 1 (with low-pass filter)')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.show()

# Spectrogram of Melody 1 Decimated (without low-pass filter - aliasing)
f2, t2, Sxx2 = scipy.signal.spectrogram(melody1alias, fs=fs_decimated, nperseg=512, noverlap=256)
plt.figure(figsize=(10, 6))
plt.pcolormesh(t2, f2, 10 * np.log10(Sxx2), shading='gouraud')
plt.title('Spectrogram - Decimated Melody 1 (without low-pass filter(alias))')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.show()

#Q17 melody1alias has distortions, as the high-frequency components are incorrectly folded back into the lower frequencies 
#melody1dec is clearer

