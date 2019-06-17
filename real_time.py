import cv2 
import time
import numpy as np
import os

from config import *

import handsegment

cap = cv2.VideoCapture(0)

'''
    0 : waiting for pressing 'space' key
    1 : preparing for 2(s)
    2 : recording for 3(s)
    3 : preprocessing
    4 : predicting 
'''

PREPARING_TIME = 2 # second
RECORDING_TIME = 4 # second  

# init state
state = 0

frames = []
reprocessed_frames = []

folder = './real_time'

if not os.path.exists(folder):
    os.mkdir(folder)

folder += '/unknown'
if not os.path.exists(folder):
    os.mkdir(folder)

predicted_label = ''

while True:

    ret, frame = cap.read()

    if not ret:
        print('Terminating...!') 
        break

    if state == 2: 
        frames.append(frame)
    elif state == 3:
        if len(frames) < FRAMES_PER_VIDEO:
           print('[WARNING] Not enough frames. At least %d frames' % (FRAMES_PER_VIDEO))
           state = 0
        else:
            frames = np.array(frames)
            ind = np.arange(0,len(frames), len(frames)/FRAMES_PER_VIDEO).astype(int)
            frames = frames[ind]
            print('index', ind)
            print('No of frames', frames.shape)

            for i  in range(len(frames)):
                f = frames[i]
                f = cv2.resize(f, SIZE)
                f = handsegment.handsegment(f)
                reprocessed_frames.append(f)
                cv2.imwrite(folder + '/unknown_frame_' + str(i) + '.jpeg', f)
            
            state = 4
            start_time = time.time()

    elif state == 4:
        # for f in reprocessed_frames:
        #     cv2.imshow('after reprocessing', f)
        #     cv2.waitKey(1)

        # PREDICT_SPATIAL 
        from predict_spatial import *
        print('[] PREDICT SPATIAL ')

        model_file = 'retrained_graph.pb'
        frames_folder = 'real_time'
        input_layer = 'Placeholder'
        output_layer = 'final_result'
        batch_size = batch_size

        train_or_test = "predict"

        # reduce tf verbosity
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        predictions = predict_on_frames(frames_folder, model_file, input_layer, output_layer, batch_size)

        out_file = 'predicted_real_time.pickle'
        print("Dumping predictions to: %s" % (out_file))
        with open(out_file, 'wb') as fout:
            pickle.dump(predictions, fout)

        print("Done.")

        # EVALUATION
        from rnn_eval import *
        print('[] EVALUATION')

        labels = load_labels('retrained_labels.txt')
        input_data_dump = 'predicted_real_time.pickle'
        num_frames_per_video = int(FRAMES_PER_VIDEO / batch_size + 1) # 201
        batch_size = batch_size
        model_file = 'non_pool.model'

        predictions = main(input_data_dump, num_frames_per_video, batch_size, labels, model_file, True)
            
        state = 0
        predicted_label = predictions[0]


            

    frame = cv2.resize(frame, (380,420))

    if state == 0:
        text = 'Press space to record %s. ' % (predicted_label)
    elif state == 1:
        text = 'Count down %.2f' % (PREPARING_TIME - (time.time() - start_time))
    elif state == 2:
        text = 'Recording %.2f' % (RECORDING_TIME - (time.time() - start_time))
    elif state == 3:
        text = 'Reprocessing %.2f' % (time.time() - start_time)
    elif state == 4:
        text = 'Predicting %.2f' % (time.time() - start_time)


    frame = cv2.putText(frame, text, (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)

    cv2.imshow('image', frame)
    key = cv2.waitKey(1)



    if key == ord(' ') and state == 0:
        state = 1
        start_time = time.time()

    elif state == 1 and (time.time() - start_time) > PREPARING_TIME:
        state = 2
        start_time = time.time()
    
    elif state == 2 and (time.time() - start_time) > RECORDING_TIME:
        state = 3
        start_time = time.time() 

    if key == ord('q'):
        break
    