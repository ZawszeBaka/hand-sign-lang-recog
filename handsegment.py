import numpy as np
import cv2
import os

def handsegment(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #lower, upper = boundaries[0]
    #lower = np.array(lower, dtype="uint8")
    #upper = np.array(upper, dtype="uint8")
    lower = np.array([0, 48, 80], dtype="uint8")
    upper = np.array([20, 255, 255], dtype="uint8")
    mask1 = cv2.inRange(hsv, lower, upper)

    #lower, upper = boundaries[1]
    lower = np.array([170,48,80], dtype="uint8")
    upper = np.array([180,255,255], dtype="uint8")
    mask2 = cv2.inRange(hsv, lower, upper)
    mask1 = mask1+mask2
    
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3),np.uint8),iterations=2)
    mask1 = cv2.dilate(mask1,np.ones((3,3),np.uint8),iterations = 1)
    mask2 = cv2.bitwise_not(mask1)


    # for i,(lower, upper) in enumerate(boundaries):
    # 	# create NumPy arrays from the boundaries
    # 	lower = np.array(lower, dtype = "uint8")
    # 	upper = np.array(upper, dtype = "uint8")

    # 	# find the colors within the specified boundaries and apply
    # 	# the mask
    # 	if(i==0):
    # 		print "Harish"
    # 		mask1 = cv2.inRange(frame, lower, upper)
    # 	else:
    # 		print "Aadi"
    # 		mask2 = cv2.inRange(frame, lower, upper)
    #mask = cv2.bitwise_or(mask1, mask2)
    output = cv2.bitwise_and(frame, frame, mask=mask1)
    # show the images
    #cv2.imshow("images", mask)
    #cv2.imshow("images", output)
    #cv2.waitKey()
    return output

def test_a_video():
    video_url = 'train_videos/Ai/AiHn2.mp4'
    cap = cv2.VideoCapture(video_url)
    ret, frame = cap.read()
    if not ret: 
        print('File %s not found' % (video_url))
    else:
        segmented_frame = handsegment(frame)
        segmented_frame = cv2.resize(segmented_frame , (420,480))
        frame = cv2.resize(frame, (420,480))
        cv2.imshow('segmented_frame', segmented_frame)
        cv2.imshow('frame', frame)
        cv2.waitKey()
        cv2.destroyAllWindows()


def test_entire_videos(dir_videos):
    labels = os.listdir(dir_videos)
    for label in labels: 
        videos_files = os.listdir(dir_videos + '/' + label) # './videos_train/Ba'
        for video_file in videos_files:

            video_url = dir_videos + '/' + label + '/' + video_file
            cap = cv2.VideoCapture(video_url) # './videos_train/Ba/'Bắt tay 3.mp4''

            print('Reading video %s' % (video_url))

            count = 0
            init = False

            ret, frame = cap.read()

            if not ret: 
                print('[ERROR] Can not read video %s' % (video_url))
            else:
                segmented_frame = handsegment(frame)
                segmented_frame = cv2.resize(segmented_frame , (420,480))
                frame = cv2.resize(frame, (420,480))
                cv2.imshow('segmented_frame', segmented_frame)
                cv2.imshow('frame', frame)
                cv2.waitKey()

    cv2.destroyAllWindows()


if __name__ == '__main__':
    # FOR TESTING HAND SEGMENTATION ONLY 
    test_entire_videos('train_videos')