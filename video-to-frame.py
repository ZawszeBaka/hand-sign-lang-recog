import cv2
import os
import pickle
from os.path import join, exists

import handsegment as hs
import argparse
import numpy as np
from tqdm import tqdm

from config import *

hc = []

def convert(gesture_folder, target_folder):
    
    if not exists('./'+target_folder):
        os.makedirs(target_folder)

    gestures = os.listdir(gesture_folder)

    # statistic
    is_init = False
    min = 0
    max = 0

    info = dict()

    for gesture in tqdm(gestures, unit='actions', ascii=True):
        
        info[gesture] = []
        videos = os.listdir(gesture_folder + '/' + gesture) # './videos_train/Ba'

        if not os.path.exists(target_folder + '/' + gesture):
            os.mkdir(target_folder + '/' + gesture)

        for video in videos:
            name = gesture_folder + '/' + gesture + '/' + video
            cap = cv2.VideoCapture(name)  # capturing input video
            property_id = int(cv2.CAP_PROP_FRAME_COUNT) 
            length = int(cv2.VideoCapture.get(cap, property_id))
            #print('Video %s has %d frames'%(video, length))

            if not is_init:
                min = length
                max = length
                is_init = True
            else:
                if length < min: 
                    min = length
                if length > max:
                    max = length

            _id = 0

            STEP = int(length / FRAMES_PER_VIDEO)
            if STEP == 0:
                continue

            t = np.arange(0, length, length/FRAMES_PER_VIDEO)
            t = t.astype(int)

            unique, counts = np.unique(t, return_counts=True)
            counts = dict(zip(unique, counts))

            saved_frames = 0
                    
            while True:
                ret, frame = cap.read()  # extract frame
                
                if ret is False:
                    info[gesture].append(saved_frames)
                    print(saved_frames)
                    break

                if np.isin(_id, t):
                    # for i in range(counts[_id]):
                    saved_frames += 1
                    frame_url = target_folder + '/' + gesture + '/' + video + "_frame_" + str(saved_frames) + ".jpeg"
                    frame = hs.handsegment(frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    frame = cv2.resize(frame, SIZE)
                    cv2.imwrite(frame_url, frame)

                _id += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                

            cap.release()

    print('STATISTIC')
    print('min = ', min)
    print('max = ', max)

    with open(gesture_folder + '_info.pickle', 'wb') as f:
        pickle.dump(info,f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract Individual Frames from gesture videos.')
    parser.add_argument('gesture_folder', help='Path to folder containing folders of videos of different gestures.')
    parser.add_argument('target_folder', help='Path to folder where extracted frames should be kept.')
    args = parser.parse_args()
    convert(args.gesture_folder, args.target_folder)
