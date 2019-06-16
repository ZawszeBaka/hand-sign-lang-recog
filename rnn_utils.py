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

def get_data(input_data_dump, num_frames_per_video, labels, ifTrain, gesture_folder):
    """Get the data from our saved predictions or pooled features."""

    # Local vars.
    X = []
    y = []
    temp_list = deque()

    with open(gesture_folder + '_info.pickle', 'rb') as f:
        info = pickle.load(f)

    pprint(info)

    # Open and get the features.
    with open(input_data_dump, 'rb') as fin:
        frames = pickle.load(fin)

        save_actual = labels[frames[0][1].lower()]
        no_frames = len(frames)
        count_label = 0

        video_id = 0
        if len(info[frames[0][1]]) > 1 :
            video_len = info[frames[0][1]][video_id] 
        else:
            video_len = 100000000000
        count_frame = 0

        print('INIT: save actual', save_actual)
        print('INIT: video len', video_len)

        for i, frame in enumerate(frames):

            features = frame[0]
            actual = frame[1] # string label

            # Convert our labels into binary.
            actual = labels[actual.lower()]

            count_label += 1
            count_frame += 1

            is_save = False
            is_clear = False

            if save_actual != actual or i == no_frames-1:              
                # for counting no of frames per label
                print('Label: ', save_actual, ', no of frames: ' , count_label)
                count_label = 0
                save_actual = actual

                # save status
                #is_save = True
                is_clear = True

                # reset video_id , switch to next video in next label
                video_id = 0
                video_len = info[frame[1]][video_id]

                # reset count frames
                count_frame = 0


            if count_frame == video_len : 
                video_id += 1
                video_len = info[frame[1]][video_id]
                if video_id == len(info[frame[1]]) - 1:
                    video_len = 1000000000 # get until end of label

                # reset count frames
                count_frame = 0

                # save status
                #is_save = True
                is_clear = True

            if count_frame == FRAMES_PER_VIDEO:
                is_save = True

            if is_save:
                # end of video
                if type(features) == list:
                    temp_list.append(features)
                flat = list(temp_list)
                X.append(np.array(flat))
                print('\n[DEBUG] shape X', np.array(temp_list).shape, ' video id', video_id , ' label', actual)
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
