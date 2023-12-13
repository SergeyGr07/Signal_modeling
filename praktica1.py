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


def amplitude_demodulation(signal):
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    return amplitude_envelope


def filter_bandpass(filter_freq, sampling_rate, noisy_fsk_signal):
    filter0 = firwin(101, filter_freq, fs=sampling_rate, pass_zero=False)
    filtered_signal0 = lfilter(filter0, 1.0, noisy_fsk_signal)

    return filtered_signal0


def phase_inverter(envelope):
    phase = np.angle(envelope)
    inverted_envelope = np.exp(1j * -phase)
    return inverted_envelope


def comparator(envelope, threshold):
    return np.where(envelope > threshold, 1, 0)


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
    plot_waveform(t, fsk_signal, 'FSK Signal')
    plot_spectrum(fsk_signal, sampling_rate, 'FSK Signal Spectrum')

    # Добавление шума к сигналу
    snr_dB = 40  # отношение сигнал-шум в децибелах
    noisy_fsk_signal = add_noise(fsk_signal, snr_dB)

    # Построение графика сигнала с шумом
    plot_waveform(t, noisy_fsk_signal, 'Noisy FSK Signal')

    # First method

    first_signal = first_method(noisy_fsk_signal)
    plot_waveform(t, first_signal, 'First Method Signal')

    # Second method
    # Фильтрация для значения 0
    filter_freq0 = 2000
    filtered_signal0 = filter_bandpass(filter_freq0, sampling_rate, noisy_fsk_signal)

    # Построение графика отфильтрованного сигнала для значения 0
    plot_waveform(t, filtered_signal0, 'Filtered Signal for 0')

    # Фильтрация для значения 1
    filter_freq1 = 6000
    filtered_signal1 = filter_bandpass(filter_freq1, sampling_rate, noisy_fsk_signal)

    # Построение графика отфильтрованного сигнала для значения 1
    plot_waveform(t, filtered_signal1, 'Filtered Signal for 1')

    # Амплитудная детекция для отфильтрованного сигнала для значения 0
    envelope_filtered0 = amplitude_demodulation(filtered_signal0)
    plot_waveform(t, envelope_filtered0, 'Amplitude Envelope for Filtered Signal 0')

    # Амплитудная детекция для отфильтрованного сигнала для значения 1
    envelope_filtered1 = amplitude_demodulation(filtered_signal1)
    plot_waveform(t, envelope_filtered1, 'Amplitude Envelope for Filtered Signal 1')

    threshold = 0.4

    comparatored_signal = comparator(envelope_filtered1, threshold)
    plot_waveform(t, comparatored_signal, 'Comparatored Signal')


if __name__ == '__main__':
    main()
