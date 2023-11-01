import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
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

# Sampling frequency (adjust based on Arduino code)
fs = 1  # Data recorded once per second

# Implement Kalman Filter
# Define the Kalman filter parameters
t_initial_state = 22.37  # Initial temperature estimate
h_initial_state = 52.55  # Initial humidity estimate
f_initial_state = 24.97  # Initial heat index estimate

initial_state_covariance = 1.0  # Initial state covariance
process_noise_covariance = 0.1  # Process noise covariance
measurement_noise_covariance = 2.0  # Measurement noise covariance

# Initialize state and state covariance
t_state = t_initial_state
h_state = h_initial_state
f_state = f_initial_state

state_covariance = initial_state_covariance

# Lists to store filtered temperature estimates and sensor readings
filtered_temperature = []
filtered_humidity = []
filtered_heat_index = []

# Kalman filtering loop for temperature
for measurement in temperature:
    # Prediction step
    state_prediction = t_state
    state_covariance_prediction = state_covariance + process_noise_covariance

    # Measurement update step
    kalman_gain = state_covariance_prediction / (state_covariance_prediction + measurement_noise_covariance)
    state = state_prediction + kalman_gain * (measurement - state_prediction)
    state_covariance = (1 - kalman_gain) * state_covariance_prediction

    filtered_temperature.append(state)

# Kalman filtering loop for humidity
for measurement in humidity:
    # Prediction step
    state_prediction = h_state
    state_covariance_prediction = state_covariance + process_noise_covariance

    # Measurement update step
    kalman_gain = state_covariance_prediction / (state_covariance_prediction + measurement_noise_covariance)
    state = state_prediction + kalman_gain * (measurement - state_prediction)
    state_covariance = (1 - kalman_gain) * state_covariance_prediction

    filtered_humidity.append(state)

# Kalman filtering loop for heat index
for measurement in heat_index:
    # Prediction step
    state_prediction = f_state
    state_covariance_prediction = state_covariance + process_noise_covariance

    # Measurement update step
    kalman_gain = state_covariance_prediction / (state_covariance_prediction + measurement_noise_covariance)
    state = state_prediction + kalman_gain * (measurement - state_prediction)
    state_covariance = (1 - kalman_gain) * state_covariance_prediction

    filtered_heat_index.append(state)

# Calculate the PSD
t_frequencies, t_psd = welch(filtered_temperature, fs=fs, nperseg=73797)
h_frequencies, h_psd = welch(filtered_humidity, fs=fs, nperseg=73797)
f_frequencies, f_psd = welch(filtered_heat_index, fs=fs, nperseg=73797)

temperature_psd = 10 * np.log10(t_psd)
humidity_psd = 10 * np.log10(h_psd)
heat_index_psd = 10 * np.log10(f_psd)

# Compute line of best fit
log_tf = np.log10(t_frequencies, out=np.zeros_like(t_frequencies), where=t_frequencies>0)
slope_t, intercept_t, r_value, p_value, std_err = linregress(log_tf, temperature_psd)
t_line = slope_t * log_tf + intercept_t

log_hf = np.log10(h_frequencies, out=np.zeros_like(h_frequencies), where=h_frequencies>0)
slope_h, intercept_h, r_value, p_value, std_err = linregress(log_hf, humidity_psd)
h_line = slope_h * log_hf + intercept_h

log_ff = np.log10(f_frequencies, out=np.zeros_like(f_frequencies), where=f_frequencies>0)
slope_f, intercept_f, r_value, p_value, std_err = linregress(log_ff, heat_index_psd)
f_line = slope_f * log_ff + intercept_f

print(slope_t, intercept_t)
print(slope_h, intercept_h)
print(slope_f, intercept_f)

# Plot the PSD
plt.figure(figsize=(10, 5))
plt.semilogx(t_frequencies, temperature_psd)
plt.plot(t_frequencies, t_line)
plt.legend(["PSD", "Line of Best Fit: -14.23 log(f) - 67.91"])
plt.title('Power Spectral Density (PSD) of Temperature Data')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB/Hz)')
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.semilogx(h_frequencies, humidity_psd)
plt.plot(h_frequencies, h_line)
plt.legend(["PSD", "Line of Best Fit: -16.32 log(f) - 64.37"])
plt.title('Power Spectral Density (PSD) of Humidity Data')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB/Hz)')
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.semilogx(f_frequencies, heat_index_psd)
plt.plot(f_frequencies, f_line)
plt.legend(["PSD", "Line of Best Fit: -14.01 log(f) - 80.30"])
plt.title('Power Spectral Density (PSD) of Heat Index Data')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (dB/Hz)')
plt.grid()
plt.show()