import numpy as np
from scipy.signal import butter, lfilter, welch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress

# Load temperature data from the CSV file
data = np.genfromtxt('arduino_data.csv', delimiter=',', skip_header=1)  # Skip the header
data_2 = np.genfromtxt('arduino_data_2.csv', delimiter=',', skip_header=1)  # Skip the header

# Extract temperature and humidity data
temperature = np.concatenate((data[1:, 0], data_2[1:, 0]))
humidity = np.concatenate((data[1:, 1], data_2[1:, 1]))

# Constants for heat index calculations
c1 = -8.78469475556
c2 = 1.61139411
c3 = 2.33854883889
c4 = -0.14611605
c5 = -0.012308094
c6 = -0.0164248277778
c7 = 2.211732 * 1e-3
c8 = 7.2546 * 1e-4
c9 = -3.582 * 1e-6

t = temperature
r = humidity
heat_index = c1 + (c2 * t) + (c3 * r) + (c4 * t * r) + (c5 * t * t) + (c6 * r * r) + (c7 * t * t * r) + (c8 * t * r * r) + (c9 * t * t * r * r)

# Normalize data
scaler = StandardScaler()
temperature_normalized = scaler.fit_transform(temperature.reshape(-1, 1))
humidity_normalized = scaler.fit_transform(humidity.reshape(-1, 1))
heat_index_normalized = scaler.fit_transform(heat_index.reshape(-1, 1))

temperature_normalized = temperature_normalized[:, 0]
humidity_normalized = humidity_normalized[:, 0]
heat_index_normalized = heat_index_normalized[:, 0]

# Sampling frequency (adjust based on Arduino code)
fs = 1  # Data recorded once per second
fc = 0.001  # Cutoff frequency in Hz 
order = 5  # Filter order

b, a = butter(order, fc, btype='low', analog=False)
filtered_temperature = lfilter(b, a, temperature_normalized)
filtered_humidity = lfilter(b, a, humidity_normalized)
filtered_heat_index = lfilter(b, a, heat_index_normalized)

# Calculate the PSD
t_frequencies, t_psd = welch(filtered_temperature, fs=fs, nperseg=35000)
h_frequencies, h_psd = welch(filtered_humidity, fs=fs, nperseg=35000)
f_frequencies, f_psd = welch(filtered_heat_index, fs=fs, nperseg=35000)

temperature_psd = 10 * np.log10(t_psd)
humidity_psd = 10 * np.log10(h_psd)
heat_index_psd = 10 * np.log10(f_psd)

# COmpute line of best fit
log_tf = np.log10(t_frequencies, out=np.zeros_like(t_frequencies), where=t_frequencies>0)
slope_t, intercept_t, r_value, p_value, std_err = linregress(log_tf, temperature_psd)
t_line = slope_t * log_tf + intercept_t

log_hf = np.log10(h_frequencies, out=np.zeros_like(h_frequencies), where=h_frequencies>0)
slope_h, intercept_h, r_value, p_value, std_err = linregress(log_hf, humidity_psd)
h_line = slope_h * log_hf + intercept_h

log_ff = np.log10(f_frequencies, out=np.zeros_like(f_frequencies), where=f_frequencies>0)
slope_f, intercept_f, r_value, p_value, std_err = linregress(log_ff, heat_index_psd)
f_line = slope_f * log_ff + intercept_f

# Plot the PSD
plt.figure(figsize=(10, 5))
plt.semilogx(t_frequencies, temperature_psd)
plt.plot(t_frequencies, t_line)
plt.legend(["PSD", "Line of Best Fit: -66.21 log(f) - 246.03"])
plt.title('Power Spectral Density (PSD) of Temperature Data')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB/Hz)')
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.semilogx(h_frequencies, humidity_psd)
plt.plot(h_frequencies, h_line)
plt.legend(["PSD", "Line of Best Fit: -67.73 log(f) - 255.01"])
plt.title('Power Spectral Density (PSD) of Humidity Data')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB/Hz)')
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.semilogx(f_frequencies, heat_index_psd)
plt.plot(f_frequencies, f_line)
plt.legend(["PSD", "Line of Best Fit: -67.29 log(f) - 255.58"])
plt.title('Power Spectral Density (PSD) of Heat Index Data')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB/Hz)')
plt.grid()
plt.show()