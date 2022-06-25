"""
Utilities for CNN based SSVEP Classification
"""
import math
import warnings

warnings.filterwarnings('ignore')

import numpy as np
from scipy.signal import butter, filtfilt

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, BatchNormalization
from keras.utils.np_utils import to_categorical
from keras import initializers, regularizers

import keras
import keras.backend as K
import keras.layers as KL
from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Activation
from keras.layers.advanced_activations import ELU
# from keras.layers import Concatenate
# from keras.layers import Add, UpSampling2D
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import DepthwiseConv2D
# from keras.layers import ZeroPadding2D
# from keras.layers import AveragePooling2D
# from keras.engine import Layer
# from keras.engine import InputSpec
# from keras.engine.topology import get_source_inputs
# from keras import backend as K
# from keras.applications import imagenet_utils
# from keras.utils import conv_utils
# from keras.optimizers import RMSprop, Adam
# from keras.utils.data_utils import get_file
from keras import regularizers


# import keras


def butter_bandpass_filter(data, lowcut, highcut, sample_rate, order):
    '''
    Returns bandpass filtered data between the frequency ranges specified in the input.

    Args:
        data (numpy.ndarray): array of samples. 
        lowcut (float): lower cutoff frequency (Hz).
        highcut (float): lower cutoff frequency (Hz).
        sample_rate (float): sampling rate (Hz).
        order (int): order of the bandpass filter.

    Returns:
        (numpy.ndarray): bandpass filtered data.
    '''

    nyq = 0.5 * sample_rate
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)
    return y


def get_filtered_eeg(eeg, lowcut, highcut, order, sample_rate):
    '''
    Returns bandpass filtered eeg for all channels and trials.

    Args:
        eeg (numpy.ndarray): raw eeg data of shape (num_classes, num_channels, num_samples, num_trials).
        lowcut (float): lower cutoff frequency (Hz).
        highcut (float): lower cutoff frequency (Hz).
        order (int): order of the bandpass filter.
        sample_rate (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): bandpass filtered eeg of shape (num_classes, num_channels, num_samples, num_trials).
    '''

    num_classes = eeg.shape[0]
    num_chan = eeg.shape[1]
    total_trial_len = eeg.shape[2]
    num_trials = eeg.shape[3]

    trial_len = int(38 + 0.135 * sample_rate + 4 * sample_rate - 1) - int(38 + 0.135 * sample_rate)
    filtered_data = np.zeros((eeg.shape[0], eeg.shape[1], trial_len, eeg.shape[3]))

    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                signal_to_filter = np.squeeze(eeg[target, channel, int(38 + 0.135 * sample_rate):
                                                                   int(38 + 0.135 * sample_rate + 4 * sample_rate - 1),
                                              trial])
                filtered_data[target, channel, :, trial] = butter_bandpass_filter(signal_to_filter, lowcut,
                                                                                  highcut, sample_rate, order)
    return filtered_data


def buffer(data, duration, data_overlap):
    '''
    Returns segmented data based on the provided input window duration and overlap.

    Args:
        data (numpy.ndarray): array of samples. 
        duration (int): window length (number of samples).
        data_overlap (int): number of samples of overlap.

    Returns:
        (numpy.ndarray): segmented data of shape (number_of_segments, duration).
    '''

    number_segments = int(math.ceil((len(data) - data_overlap) / (duration - data_overlap)))
    temp_buf = [data[i:i + duration] for i in range(0, len(data), (duration - int(data_overlap)))]
    temp_buf[number_segments - 1] = np.pad(temp_buf[number_segments - 1],
                                           (0, duration - temp_buf[number_segments - 1].shape[0]),
                                           'constant')
    segmented_data = np.vstack(temp_buf[0:number_segments])

    return segmented_data


def get_segmented_epochs(data, window_len, shift_len, sample_rate):
    '''
    Returns epoched eeg data based on the window duration and step size.

    Args:
        data (numpy.ndarray): array of samples. 
        window_len (int): window length (seconds).
        shift_len (int): step size (seconds).
        sample_rate (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): epoched eeg data of shape. 
        (num_classes, num_channels, num_trials, number_of_segments, duration).
    '''

    num_classes = data.shape[0]
    num_chan = data.shape[1]
    num_trials = data.shape[3]

    duration = int(window_len * sample_rate)
    data_overlap = (window_len - shift_len) * sample_rate

    number_of_segments = int(math.ceil((data.shape[2] - data_overlap) /
                                       (duration - data_overlap)))

    segmented_data = np.zeros((data.shape[0], data.shape[1],
                               data.shape[3], number_of_segments, duration))

    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                segmented_data[target, channel, trial, :, :] = buffer(data[target, channel, :, trial],
                                                                      duration, data_overlap)

    return segmented_data


def magnitude_spectrum_features(segmented_data, FFT_PARAMS):
    '''
    Returns magnitude spectrum features. Fast Fourier Transform computed based on
    the FFT parameters provided as input.

    Args:
        segmented_data (numpy.ndarray): epoched eeg data of shape 
        (num_classes, num_channels, num_trials, number_of_segments, num_samples).
        FFT_PARAMS (dict): dictionary of parameters used for feature extraction.
        FFT_PARAMS['resolution'] (float): frequency resolution per bin (Hz).
        FFT_PARAMS['start_frequency'] (float): start frequency component to pick from (Hz). 
        FFT_PARAMS['end_frequency'] (float): end frequency component to pick upto (Hz). 
        FFT_PARAMS['sampling_rate'] (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): magnitude spectrum features of the input EEG.
        (n_fc, num_channels, num_classes, num_trials, number_of_segments).
    '''

    num_classes = segmented_data.shape[0]
    num_chan = segmented_data.shape[1]
    num_trials = segmented_data.shape[2]
    number_of_segments = segmented_data.shape[3]
    fft_len = segmented_data[0, 0, 0, 0, :].shape[0]

    NFFT = round(FFT_PARAMS['sampling_rate'] / FFT_PARAMS['resolution'])
    fft_index_start = int(round(FFT_PARAMS['start_frequency'] / FFT_PARAMS['resolution']))
    fft_index_end = int(round(FFT_PARAMS['end_frequency'] / FFT_PARAMS['resolution'])) + 1

    features_data = np.zeros(((fft_index_end - fft_index_start),
                              segmented_data.shape[1], segmented_data.shape[0],
                              segmented_data.shape[2], segmented_data.shape[3]))

    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                for segment in range(0, number_of_segments):
                    temp_FFT = np.fft.fft(segmented_data[target, channel, trial, segment, :], NFFT) / fft_len
                    magnitude_spectrum = 2 * np.abs(temp_FFT)
                    features_data[:, channel, target, trial, segment] = magnitude_spectrum[
                                                                        fft_index_start:fft_index_end, ]

    return features_data


def complex_spectrum_features(segmented_data, FFT_PARAMS):
    '''
    Returns complex spectrum features. Fast Fourier Transform computed based on
    the FFT parameters provided as input. The real and imaginary parts of the input
    signal are concatenated into a single feature vector.

    Args:
        segmented_data (numpy.ndarray): epoched eeg data of shape 
        (num_classes, num_channels, num_trials, number_of_segments, num_samples).
        FFT_PARAMS (dict): dictionary of parameters used for feature extraction.
        FFT_PARAMS['resolution'] (float): frequency resolution per bin (Hz).
        FFT_PARAMS['start_frequency'] (float): start frequency component to pick from (Hz). 
        FFT_PARAMS['end_frequency'] (float): end frequency component to pick upto (Hz). 
        FFT_PARAMS['sampling_rate'] (float): sampling rate (Hz).

    Returns:
        (numpy.ndarray): complex spectrum features of the input EEG.
        (2*n_fc, num_channels, num_classes, num_trials, number_of_segments)
    '''

    num_classes = segmented_data.shape[0]
    num_chan = segmented_data.shape[1]
    num_trials = segmented_data.shape[2]
    number_of_segments = segmented_data.shape[3]
    fft_len = segmented_data[0, 0, 0, 0, :].shape[0]

    NFFT = round(FFT_PARAMS['sampling_rate'] / FFT_PARAMS['resolution'])
    fft_index_start = int(round(FFT_PARAMS['start_frequency'] / FFT_PARAMS['resolution']))
    fft_index_end = int(round(FFT_PARAMS['end_frequency'] / FFT_PARAMS['resolution'])) + 1

    features_data = np.zeros((2 * (fft_index_end - fft_index_start),
                              segmented_data.shape[1], segmented_data.shape[0],
                              segmented_data.shape[2], segmented_data.shape[3]))

    for target in range(0, num_classes):
        for channel in range(0, num_chan):
            for trial in range(0, num_trials):
                for segment in range(0, number_of_segments):
                    temp_FFT = np.fft.fft(segmented_data[target, channel, trial, segment, :], NFFT) / fft_len
                    real_part = np.real(temp_FFT)
                    imag_part = np.imag(temp_FFT)
                    features_data[:, channel, target, trial, segment] = np.concatenate((
                        real_part[fft_index_start:fft_index_end, ],
                        imag_part[fft_index_start:fft_index_end, ]), axis=0)

    return features_data


# CAM
def channel_attention(input_xs, reduction_ratio=0.5):
    # get channel
    # channel = int(input_xs.shape[channel_axis])
    # 判断输入数据格式，是channels_first还是channels_last
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1

    channel_axis = 3
    channel = int(input_xs.shape[channel_axis])
    maxpool_channel = KL.GlobalMaxPooling2D()(input_xs)
    avgpool_channel = KL.GlobalAvgPool2D()(input_xs)
    Dense_One = KL.Dense(units=int(channel * reduction_ratio),
                         activation='relu',
                         kernel_initializer='he_normal',
                         use_bias=True,
                         bias_initializer='zeros')
    Dense_Two = KL.Dense(units=int(channel),
                         activation='relu',
                         use_bias=True,
                         bias_initializer='zeros')
    # 最大池化通道
    mlp_1_max = Dense_One(maxpool_channel)
    mlp_2_max = Dense_Two(mlp_1_max)
    mlp_2_max = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_max)
    # 平均池化通道
    mlp_1_avg = Dense_One(avgpool_channel)
    mlp_2_avg = Dense_Two(mlp_1_avg)
    mlp_2_avg = KL.Reshape(target_shape=(1, 1, int(channel)))(mlp_2_avg)
    channel_attention_feature = KL.Add()([mlp_2_max, mlp_2_avg])
    channel_attention_feature = Activation('sigmoid')(channel_attention_feature)
    channel_attention_feature = KL.Multiply()([channel_attention_feature, input_xs])
    return channel_attention_feature


# SAM
def spatial_attention(channel_refined_feature):
    maxpool_spatial = KL.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(channel_refined_feature)
    avgpool_spatial = KL.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(channel_refined_feature)
    max_avg_pool_spatial = KL.Concatenate(axis=3)([maxpool_spatial, avgpool_spatial])
    spatial_attention_feature = KL.Conv2D(filters=1,
                                          kernel_size=(1, 3),
                                          padding="same",
                                          strides=1,
                                          activation='sigmoid',
                                          kernel_initializer='he_normal',
                                          use_bias=False)(max_avg_pool_spatial)
    return spatial_attention_feature


# CBAM
def cbam_module(input_xs, reduction_ratio=0.5):
    channel_refined_feature = channel_attention(input_xs, reduction_ratio=reduction_ratio)
    spatial_attention_feature = spatial_attention(channel_refined_feature)
    refined_feature = KL.Multiply()([channel_refined_feature, spatial_attention_feature])
    refined_feature = KL.Add()([refined_feature, input_xs])
    return refined_feature


def CNN_model(input_shape, CNN_PARAMS):
    img_input = Input(shape=input_shape)

    x = Conv2D(2 * CNN_PARAMS['n_ch'],
               kernel_size=(CNN_PARAMS['n_ch'], 1),
               input_shape=(input_shape[0], input_shape[1], input_shape[2]),
               padding="valid",
               kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']),
               kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(CNN_PARAMS['droprate'])(x)

    x = cbam_module(x)

    x = Conv2D(2 * CNN_PARAMS['n_ch'],
               kernel_size=(1, CNN_PARAMS['kernel_f']),
               kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']),
               padding="valid",
               kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(CNN_PARAMS['droprate'])(x)

    x = cbam_module(x)

    class_x = layers.Flatten()(x)
    cls_out = Dense(CNN_PARAMS['num_classes'], activation='softmax',
          kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']),
          kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None),name="cls_out")(class_x)

    model = Model(img_input, cls_out)

    return model


def CNN_model2(input_shape, CNN_PARAMS):
    model = Sequential()
    model.add(Conv2D(2 * CNN_PARAMS['n_ch'], kernel_size=(CNN_PARAMS['n_ch'], 1),
                     input_shape=(input_shape[0], input_shape[1], input_shape[2]),
                     padding="valid", kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']),
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(CNN_PARAMS['droprate']))
    model.add(Conv2D(2 * CNN_PARAMS['n_ch'], kernel_size=(1, CNN_PARAMS['kernel_f']),
                     kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), padding="valid",
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(CNN_PARAMS['droprate']))
    model.add(Flatten())
    model.add(Dense(CNN_PARAMS['num_classes'], activation='softmax',
                    kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']),
                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))

    return model

    '''
    Returns the Concolutional Neural Network model for SSVEP classification.

    Args:
        input_shape (numpy.ndarray): shape of input training data 
        e.g. [num_training_examples, num_channels, n_fc] or [num_training_examples, num_channels, 2*n_fc].
        CNN_PARAMS (dict): dictionary of parameters used for feature extraction.        
        CNN_PARAMS['batch_size'] (int): training mini batch size.
        CNN_PARAMS['epochs'] (int): total number of training epochs/iterations.
        CNN_PARAMS['droprate'] (float): dropout ratio.
        CNN_PARAMS['learning_rate'] (float): model learning rate.
        CNN_PARAMS['lr_decay'] (float): learning rate decay ratio.
        CNN_PARAMS['l2_lambda'] (float): l2 regularization parameter.
        CNN_PARAMS['momentum'] (float): momentum term for stochastic gradient descent optimization.
        CNN_PARAMS['kernel_f'] (int): 1D kernel to operate on conv_1 layer for the SSVEP CNN. 
        CNN_PARAMS['n_ch'] (int): number of eeg channels
        CNN_PARAMS['num_classes'] (int): number of SSVEP targets/classes

    Returns:
        (keras.Sequential): CNN model.
    '''

    '''
    model = Sequential()
    model.add(Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(CNN_PARAMS['n_ch'], 1), 
                     input_shape=(input_shape[0], input_shape[1], input_shape[2]), 
                     padding="valid", kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), 
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(CNN_PARAMS['droprate']))  
    model.add(Conv2D(2*CNN_PARAMS['n_ch'], kernel_size=(1, CNN_PARAMS['kernel_f']), 
                     kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), padding="valid", 
                     kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(CNN_PARAMS['droprate']))  
    model.add(Flatten())
    model.add(Dense(CNN_PARAMS['num_classes'], activation='softmax', 
                    kernel_regularizer=regularizers.l2(CNN_PARAMS['l2_lambda']), 
                    kernel_initializer=initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)))
    
    return model
    '''
