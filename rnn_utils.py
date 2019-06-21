"""
Utilities used by our other RNN scripts.
"""
from collections import deque
from sklearn.model_selection import train_test_split
from tflearn.data_utils import to_categorical
import tflearn
import numpy as np
import pickle
from pprint import pprint

from config import *

def get_data(input_data_dump, num_frames_per_video, labels, ifTrain, gesture_folder, is_predict):
    """Get the data from our saved predictions or pooled features."""

    # Local vars.
    X = []
    y = []
    temp_list = deque()

    NUM_DATA_PER_VIDEOS = 10 # int(FRAMES_PER_VIDEO / batch_size + 1)

    # Open and get the features.
    with open(input_data_dump, 'rb') as fin:
        frames = pickle.load(fin)

        print('DEBUG: len frames', len(frames))
        print('LABELS', labels)

        if not is_predict:
            save_actual = labels[frames[0][1].lower()]
            no_frames = len(frames)
            count_label = 0
            print('INIT: save actual', save_actual)

        count_frame = 0

        for i, frame in enumerate(frames):

            features = frame[0]
            actual = frame[1] # string label

            # Convert our labels into binary.
            if not is_predict:
                actual = labels[actual.lower()]

            # count_label += 1
            count_frame += 1

            is_save = False
            is_clear = False

            if count_frame == NUM_DATA_PER_VIDEOS:
                is_save = True
                is_clear = True
                count_frame = 0

            if is_save:
                # end of video
                if type(features) == list:
                    temp_list.append(features)
                flat = list(temp_list)
                if len(flat) == NUM_DATA_PER_VIDEOS:
                    X.append(np.array(flat))
                    if not is_predict:
                        print('\n[DEBUG] shape X', np.array(temp_list).shape, ' label', actual)
                    #X.append(np.array(temp_list))
                    # pprint(temp_list)
                    y.append(actual)
            else:
                if type(features) == list:
                    temp_list.append(features)

            if is_clear:
                temp_list.clear()

    print("Class Name\tNumeric Label")
    for key in labels:
        print("%s\t\t%d" % (key, labels[key]))

    # Numpy.
    X = np.array(X)
    y = np.array(y)

    print("Dataset shape: ", X.shape)
    print("y shape: ", y.shape)

    # One-hot encoded categoricals.
    if not is_predict:
        y = to_categorical(y, len(labels))

    # Split into train and test.
    if ifTrain:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_test, y_train, y_test
    else:
        return X, y


def get_network(frames, input_size, num_classes):
    """Create our LSTM"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    net = tflearn.lstm(net, 128, dropout=0.8, return_seq=True)
    net = tflearn.lstm(net, 128)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name="output1")
    return net


def get_network_deep(frames, input_size, num_classes):
    """Create a deeper LSTM"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    #net = tflearn.input_data(shape=[None, None, input_size])
    net = tflearn.lstm(net, 64, dropout=0.2, return_seq=True)
    net = tflearn.lstm(net, 64, dropout=0.2, return_seq=True)
    net = tflearn.lstm(net, 64, dropout=0.2)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name="output1")
    return net


def get_network_wide(frames, input_size, num_classes):
    """Create a wider LSTM"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    net = tflearn.lstm(net, 256, dropout=0.2)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name='output1')
    return net


def get_network_wider(frames, input_size, num_classes):
    """Create a wider LSTM"""
    net = tflearn.input_data(shape=[None, frames, input_size])
    net = tflearn.lstm(net, 512, dropout=0.2)
    net = tflearn.fully_connected(net, num_classes, activation='softmax')
    net = tflearn.regression(net, optimizer='adam',
                             loss='categorical_crossentropy', name='output1')
    return net
