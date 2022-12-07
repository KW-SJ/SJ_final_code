# Deep Physiological Sensing Toolbox
# Xin Liu, Xiaoyu Zhang, Girish Narayanswamy, Yuzhe Zhang, Yuntao Wang, Shwetak Patel, Daniel McDuff
# https://arxiv.org/abs/2210.00716


import numpy as np
import scipy
from scipy.signal import butter
from scipy.sparse import spdiags


def process_video(frames):
    RGB = []
    for frame in frames:
        summation = np.sum(np.sum(frame, axis=0), axis=0)
        RGB.append(summation / (frame.shape[0] * frame.shape[1]))
    RGB = np.asarray(RGB)
    RGB = RGB.transpose(1, 0).reshape(1, 3, -1)
    return np.asarray(RGB)


def calculate_HR(pxx_pred, frange_pred, fmask_pred):
    pred_HR = np.take(frange_pred, np.argmax(
        np.take(pxx_pred, fmask_pred), 0))[0] * 60
    return pred_HR


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def detrend(input_signal, lambda_value):
    signal_length = input_signal.shape[0]
    # observation matrix
    H = np.identity(signal_length)
    ones = np.ones(signal_length)
    minus_twos = -2 * np.ones(signal_length)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index,
                (signal_length - 2), signal_length).toarray()
    filtered_signal = np.dot(
        (H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D))), input_signal)
    return filtered_signal


def calculate_metric_per_video(predictions, labels, diff_flag=True, signal='pulse', fs=30, bpFlag=True):
    if signal == 'pulse':
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2],
                        btype='bandpass')  # 2.5 -> 1.7
    else:
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')

    if diff_flag:
        if signal == 'pulse':
            pred_window = detrend(np.cumsum(predictions), 100)
            # label_window = detrend(np.cumsum(labels), 100)
        else:
            pred_window = np.cumsum(predictions)
    else:
        if signal == 'pulse':
            pred_window = detrend(predictions, 100)
            # label_window = detrend(labels, 100)
        else:
            pred_window = predictions

    if bpFlag:
        pred_window = scipy.signal.filtfilt(b, a, np.double(pred_window))
        # label_window = scipy.signal.filtfilt(b, a, np.double(label_window))

    pred_window = np.expand_dims(pred_window, 0)
    # label_window = np.expand_dims(label_window, 0)
    # Predictions FFT
    N = next_power_of_2(pred_window.shape[1])
    f_prd, pxx_pred = scipy.signal.periodogram(
        pred_window, fs=fs, nfft=N, detrend=False)
    if signal == 'pulse':
        # regular Heart beat are 0.75*60 and 2.5*60
        fmask_pred = np.argwhere((f_prd >= 0.75) & (f_prd <= 2.5))
    else:
        # regular Heart beat are 0.75*60 and 2.5*60
        fmask_pred = np.argwhere((f_prd >= 0.08) & (f_prd <= 0.5))
    pred_window = np.take(f_prd, fmask_pred)
    
    # # Labels FFT
    # f_label, pxx_label = scipy.signal.periodogram(
    #     label_window, fs=fs, nfft=N, detrend=False)
    # if signal == 'pulse':
    #     # regular Heart beat are 0.75*60 and 2.5*60
    #     fmask_label = np.argwhere((f_label >= 0.75) & (f_label <= 2.5))
    # else:
    #     # regular Heart beat are 0.75*60 and 2.5*60
    #     fmask_label = np.argwhere((f_label >= 0.08) & (f_label <= 0.5))
    # label_window = np.take(f_label, fmask_label)

    # MAE
    temp_HR = calculate_HR(
        pxx_pred, pred_window, fmask_pred)
    return temp_HR


def calculate_metric_peak_per_video(predictions, labels, diff_flag=True, signal='pulse', window_size=100, fs=30,
                                    bpFlag=True):
    if signal == 'pulse':
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2],
                        btype='bandpass')  # 2.5 -> 1.7
    else:
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')

    data_len = len(predictions)
    HR_pred = []
    # HR0_pred = []
    all_peaks = []
    # all_peaks0 = []
    # pred_signal = []
    # label_signal = []
    window_size = data_len
    for j in range(0, data_len, window_size):
        if j == 0 and (j + window_size) > data_len:
            pred_window = predictions
            # label_window = labels
        elif (j + window_size) > data_len:
            break
        else:
            pred_window = predictions[j:j + window_size]
            # label_window = labels[j:j + window_size]
        if diff_flag:
            if signal == 'pulse':
                pred_window = detrend(np.cumsum(pred_window), 100)
                # label_window = detrend(np.cumsum(label_window), 100)
            else:
                pred_window = np.cumsum(pred_window)
        else:
            if signal == 'pulse':
                pred_window = detrend(pred_window, 100)
                # label_window = detrend(label_window, 100)
            else:
                pred_window = pred_window

        if bpFlag:
            pred_window = scipy.signal.filtfilt(b, a, pred_window)
            # label_window = scipy.signal.filtfilt(b, a, label_window)

        # Peak detection
        # labels_peaks, _ = scipy.signal.find_peaks(label_window)
        preds_peaks, _ = scipy.signal.find_peaks(pred_window)

        # temp_HR_0 = 60 / (np.mean(np.diff(labels_peaks)) / fs)
        temp_HR = 60 / (np.mean(np.diff(preds_peaks)) / fs)

        HR_pred.append(temp_HR)
        # HR0_pred.append(temp_HR_0)
        all_peaks.extend(preds_peaks + j)
        # all_peaks0.extend(labels_peaks + j)
        # pred_signal.extend(pred_window.tolist())
        # label_signal.extend(label_window.tolist())

    HR = np.mean(np.array(HR_pred))
    # HR0 = np.mean(np.array(HR0_pred))

    return HR