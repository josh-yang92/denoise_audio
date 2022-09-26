from ast import parse
import os
import sys
import argparse

import scipy.io.wavfile
from scipy.fft import rfft, rfftfreq, irfft
import numpy as np

def normalise(data):
    return data/32768


def process(signal, N, sample_rate, filter_frequencies):
    yf = rfft(signal)
    xf = rfftfreq(N, 1 / sample_rate)

    points_per_freq = len(xf) / (sample_rate / 2)
    print(f'points per freq: {points_per_freq}')

    target_idx = []
    for freq in filter_frequencies:
        target_idx.append(int(points_per_freq * float(freq)))

    print(f'target indexes: {target_idx}')

    yf_copy = yf.copy()

    # we filter 3 frequencies below & above the chosen frequency. If you want to filter out the exact freqencies
    # then change the filter_range to 1. i.e. filter_range = 1
    filter_range = 5 if int(points_per_freq) == 0 else int(points_per_freq)*3
    for i, idx in enumerate(target_idx):
        if filter_range == 1:
            print(f'filtering indexes: {idx}')
            yf_copy[idx] = 0
            continue
        print(f'filtering indexes: {idx-filter_range}: {idx+filter_range}')
        if i == 0:
            yf_copy[idx: idx+filter_range] = 0
        else:
            yf_copy[idx-filter_range: idx+filter_range] = 0

    return yf_copy

def parse_arguments():
    parser = argparse.ArgumentParser(description='audio (.wav) denoising')
    parser.add_argument('root_dir', metavar='root_dir', type=str,
                    help='root directory where the .wav file resides.')
    parser.add_argument('file_name', metavar='file_name', type=str,
                    help='.wav file name without the extension.')
    parser.add_argument('--freq', default=[60], nargs='+', help='list of frequencies you want to filter out. Default = 60. Input example: 60 120')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_arguments()
    
    rate, data = scipy.io.wavfile.read(os.path.join(args.root_dir, f'{args.file_name}.wav'))
    print('-'*100)
    print('audio file loaded!')

    no_samples = len(data)

    normalised_data = normalise(data)
    new_sig = np.empty_like(normalised_data)
    print('-'*100)
    print('starting processing...')
    if len(normalised_data.shape) == 1:
        print(f'processing...')
        processed = process(normalised_data, filter_frequencies=args.freq, N=no_samples, sample_rate=rate)
        inversefft_processed = irfft(processed)
        shape_diff = inversefft_processed.shape[0] - new_sig.shape[0]

        if shape_diff <= 0:
            new_sig[:] = np.hstack([inversefft_processed, np.zeros(shape_diff)])
        elif shape_diff >= 0:
            inversefft_processed = np.delete(inversefft_processed, (shape_diff), axis=0)
            new_sig[:] = inversefft_processed
        else:
            new_sig[:] = inversefft_processed
    else:
        for i in range(normalised_data.shape[1]):
            print(f'processing channel {i}...')
            processed = process(normalised_data[:,i], filter_frequencies=args.freq, N=no_samples, sample_rate=rate)
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
    output_path = os.path.join(args.root_dir, f'{args.file_name}_processed.wav')
    scipy.io.wavfile.write(output_path, rate, new_sig)
    print(f'done. saved in {output_path}')