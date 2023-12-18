import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, lfilter, firwin, filtfilt, butter


def generate_fsk_signal(binary_sequence, bit_rate, carrier_freq1, carrier_freq2, sampling_rate):
    t = np.arange(0, len(binary_sequence) / bit_rate, 1 / sampling_rate)
    signal = np.zeros_like(t)

    for i, bit in enumerate(binary_sequence):
        if bit == 0:
            signal[i * int(sampling_rate / bit_rate):(i + 1) * int(sampling_rate / bit_rate)] = np.sin(
                2 * np.pi * carrier_freq1 * t[i * int(sampling_rate / bit_rate):(i + 1) * int(sampling_rate / bit_rate)])
        else:
            signal[i * int(sampling_rate / bit_rate):(i + 1) * int(sampling_rate / bit_rate)] = np.sin(
                2 * np.pi * carrier_freq2 * t[i * int(sampling_rate / bit_rate):(i + 1) * int(sampling_rate / bit_rate)])

    return t, signal


def add_noise(signal, snr_dB):
    noise_power = 10**(-snr_dB / 10)
    noise = np.sqrt(noise_power) * np.random.normal(0, 1, len(signal))
    noisy_signal = signal + noise
    return noisy_signal


def plot_waveform(t, signal, title):
    plt.figure(figsize=(12, 6))
    plt.plot(t, signal)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()


def plot_spectrum(signal, sampling_rate, title):
    plt.figure(figsize=(12, 6))
    plt.magnitude_spectrum(signal, Fs=sampling_rate, scale='dB', color='C1')
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (dB)')
    plt.grid(True)
    plt.show()


def first_method(fsk_signal):
    sign_signal = np.sign(fsk_signal)
    not_minus_sugn_signal = []
    for element in sign_signal:
        not_minus_sugn_signal.append(element + 1)

    final_signal = []
    found_first_one = False

    for digit in not_minus_sugn_signal:
        if digit == 2.0:
            if not found_first_one:
                final_signal.append(2.0)
                found_first_one = True
            else:
                final_signal.append(0.0)
        elif digit == 0.0:
            found_first_one = False
            final_signal.append(0.0)

    b, a = butter(8, 0.02, 'lowpass')
    filtedData = filtfilt(b, a, final_signal)

    result = []

    for sample in filtedData:
        # Сравниваем амплитуду с пороговым значением
        if sample > 0.08:
            result.append(1)  # Логическая единица
        else:
            result.append(0)  # Логический ноль

    return result


def third_method(t, fsk_signal):
    ones_cos = np.cos(2 * np.pi * 6000 * t - (np.pi) / 2)
    zeros_cos = np.cos(2 * np.pi * 2000 * t - (np.pi) / 2)

    zeros_signal = fsk_signal * zeros_cos
    ones_signal = fsk_signal * ones_cos

    filt_zeros = butter_lowpass_filter(zeros_signal, 2000, 100000)

    filt_ones = butter_lowpass_filter(ones_signal, 2000, 100000)

    total_signal = (filt_ones - filt_zeros) + 0.8

    comp_sign = comparator(total_signal, 0.9)

    return comp_sign


def amplitude_demodulation(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope


def filter_bandpass(filter_freq, sampling_rate, noisy_fsk_signal):
    filter0 = firwin(101, filter_freq, fs=sampling_rate, pass_zero=False)
    filtered_signal0 = lfilter(filter0, 1.0, noisy_fsk_signal)

    return filtered_signal0


def comparator(envelope, threshold):
    return np.where(envelope > threshold, 1, 0)


def correlation_demodulation(carrier_freq, noisy_fsk_signal, sampling_rate):
    signal_length = len(noisy_fsk_signal)

    carrier_signal = np.sin(2 * np.pi * carrier_freq * np.arange(signal_length) / sampling_rate)

    correlated_signal = noisy_fsk_signal * carrier_signal
    # correlated_signal = filter_bandpass(2000, sampling_rate, correlated_signal)
    # filtered_signal = butter_lowpass_filter(noisy_fsk_signal, 2000, sampling_rate)
    # Интегратор
    integrator_function = np.cumsum(correlated_signal) / sampling_rate

    # direvative_function = np.
    derivative_function = np.gradient(integrator_function, 1 / sampling_rate)

    # Выделить рост, постоянные сместить к нулю, сделать пилообразный
    growth_signal = np.where(derivative_function > 0, derivative_function, 0)
    constant_shifted_signal = integrator_function - np.mean(integrator_function)
    sawtooth_signal = constant_shifted_signal - growth_signal

    threshold = 0.4

    demodulated_signal = comparator(sawtooth_signal, threshold)

    return demodulated_signal, correlated_signal


def butter_lowpass_filter(data, cutoff_freq, sampling_rate, order=5):
    nyquist = 0.5 * sampling_rate
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data


def butter_bandpass_filter(data, low_cutoff, high_cutoff, sampling_rate, order=5):
    nyquist = 0.5 * sampling_rate
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(N=6, Wn=[low, high], btype='band')
    filtered_signal = lfilter(b, a, data)

    return filtered_signal


# infradyne
def superheterodyne_demodulation(carrier_freq, noisy_fsk_signal, sampling_rate):
    # Определение длины сигнала.
    signal_length = len(noisy_fsk_signal)

    # Создание сигнала опорной частоты.
    carrier_signal = np.sin(2 * np.pi * carrier_freq * np.arange(signal_length) / sampling_rate)

    # Перемножение сигнала опорной частоты и шумового FSK сигнала.
    correlated_signal = noisy_fsk_signal * carrier_signal

    # Фильтрация полосы пропускания.
    # Добавление полосового фильтра от 21000 до 27000 Гц.
    low_cutoff = 21000
    high_cutoff = 27000
    filtered_signal = butter_bandpass_filter(correlated_signal, low_cutoff, high_cutoff, sampling_rate)

    low_cutoff = 24000
    high_cutoff = 28000

    filtered_signal = butter_bandpass_filter(filtered_signal, low_cutoff, high_cutoff, sampling_rate)

    # Амплитудная детекция.
    envelope_signal = amplitude_demodulation(filtered_signal)

    # Пороговое детектирование.
    threshold = 0.4
    demodulated_signal = comparator(envelope_signal, threshold)

    return demodulated_signal, filtered_signal, noisy_fsk_signal


def main():
    # Параметры сигнала
    bit_rate = 400  # битрейт
    carrier_freq1 = 2000  # частота первого несущего сигнала
    carrier_freq2 = 6000  # частота второго несущего сигнала
    sampling_rate = 100000  # частота дискретизации

    # Генерация двоичной последовательности
    binary_sequence = [1, 0, 1, 0, 1, 0]

    # Генерация FSK сигнала
    t, fsk_signal = generate_fsk_signal(binary_sequence, bit_rate, carrier_freq1, carrier_freq2, sampling_rate)

    # Построение осциллограммы и спектра FSK сигнала
    # plot_waveform(t, fsk_signal, 'FSK Signal')
    # plot_spectrum(fsk_signal, sampling_rate, 'FSK Signal Spectrum')

    # Добавление шума к сигналу
    snr_dB = 40  # отношение сигнал-шум в децибелах
    noisy_fsk_signal = add_noise(fsk_signal, snr_dB)

    # Построение графика сигнала с шумом
    # plot_waveform(t, noisy_fsk_signal, 'Noisy FSK Signal')

    # First method

    first_signal = first_method(noisy_fsk_signal)
    # plot_waveform(t, first_signal, 'First Method Signal')

    # Second method
    # Фильтрация для значения 0
    filter_freq0 = 2000
    filtered_signal0 = filter_bandpass(filter_freq0, sampling_rate, noisy_fsk_signal)

    # Построение графика отфильтрованного сигнала для значения 0
    # plot_waveform(t, filtered_signal0, 'Filtered Signal for 0')

    # Фильтрация для значения 1
    filter_freq1 = 6000
    filtered_signal1 = filter_bandpass(filter_freq1, sampling_rate, noisy_fsk_signal)

    # Построение графика отфильтрованного сигнала для значения 1
    # plot_waveform(t, filtered_signal1, 'Filtered Signal for 1')

    # Амплитудная детекция для отфильтрованного сигнала для значения 0
    envelope_filtered0 = amplitude_demodulation(filtered_signal0)
    # plot_waveform(t, envelope_filtered0, 'Amplitude Envelope for Filtered Signal 0')

    # Амплитудная детекция для отфильтрованного сигнала для значения 1
    envelope_filtered1 = amplitude_demodulation(filtered_signal1)
    # plot_waveform(t, envelope_filtered1, 'Amplitude Envelope for Filtered Signal 1')

    threshold = 0.4

    comparatored_signal = comparator(envelope_filtered1, threshold)
    # plot_waveform(t, comparatored_signal, 'Comparatored Signal')

    # Fourth method
    # Корреляционная демодуляция.
    demodulated_signal, correlated_signal = correlation_demodulation(carrier_freq2, noisy_fsk_signal, sampling_rate)
    plot_waveform(t, correlated_signal, "correlated_signal1")

    # # Построение графика демодулированного сигнала.
    plot_waveform(t, demodulated_signal, 'Correlation Demodulation Signal')

    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.plot(noisy_fsk_signal, label='Correlated Signal')
    plt.title('Correlated Signal')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(demodulated_signal, label='Demodulated Signal', color='orange')
    plt.title('Demodulated Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    plot_spectrum(demodulated_signal, sampling_rate, "correlated_signal")

    # # Fivths method
    # # Корреляционная демодуляция.
    carrier_freq_infradyne = 20000
    demodulated_signal, correlated_signal, noisy_fsk_signal = superheterodyne_demodulation(carrier_freq_infradyne, noisy_fsk_signal, sampling_rate)

    # # Построение графика демодулированного сигнала.
    plot_waveform(t, demodulated_signal, 'Superheterodyne Demodulation Signal')
    # plot_waveform(t, correlated_signal, "correlated_signal")
    # plot_spectrum(correlated_signal, sampling_rate, "correlated_signal")
    # plot_waveform(t, noisy_fsk_signal, "correlated_signal")


if __name__ == '__main__':
    main()
