import os
import sys

import scipy.io.wavfile
from scipy.fft import rfft, rfftfreq, irfft
import numpy as np

def normalise(data):
    return data/32768


def process(signal, N, sample_rate):
    yf = rfft(signal)
    xf = rfftfreq(N, 1 / sample_rate)

    points_per_freq = len(xf) / (sample_rate / 2)
    print(f'points per freq: {points_per_freq}')

    target_idx = []
    for i in range(0, 500, 50):
        target_idx.append(int(points_per_freq * i))
    target_idx_20000 = int(points_per_freq * 20000)

    print(f'target indexes: {target_idx}')

    yf_copy = yf.copy()

    filter_range = 5 if int(points_per_freq) == 0 else int(points_per_freq)*2
    for i, idx in enumerate(target_idx):
        print(f'filtering indexes: {idx-filter_range}: {idx+filter_range}')
        if i == 0:
            yf_copy[idx: idx+filter_range] = 0
        else:
            yf_copy[idx-filter_range: idx+filter_range] = 0
    yf_copy[target_idx_20000:] = 0

    return yf_copy


if __name__ == '__main__':
    directory = sys.argv[1]
    file_name = sys.argv[2]

    rate, data = scipy.io.wavfile.read(os.path.join(directory, f'{file_name}.wav'))
    print('-'*100)
    print('audio file loaded!')

    no_samples = len(data)
    sampled_data = data[0:no_samples,:]

    normalised_data = normalise(sampled_data)

    new_sig = np.empty_like(normalised_data)
    print('-'*100)
    print('starting processing...')
    for i in range(normalised_data.shape[1]):
        print(f'processing channel {i}...')
        processed = process(normalised_data[:,i], N=no_samples, sample_rate=rate)
        inversefft_processed = irfft(processed)
        shape_diff = inversefft_processed.shape[0] - new_sig.shape[0]

        if shape_diff <= 0:
            new_sig[:,i] = np.hstack([inversefft_processed, np.zeros(shape_diff)])
        elif shape_diff >= 0:
            inversefft_processed = np.delete(inversefft_processed, (shape_diff), axis=0)
            new_sig[:,i] = inversefft_processed
        else:
            new_sig[:,i] = inversefft_processed
    
    print('-'*100)
    print('saving processed audio file...')
    output_path = os.path.join(directory, f'{file_name}_processed.wav')
    scipy.io.wavfile.write(output_path, rate, new_sig)
    print(f'done. saved in {output_path}')