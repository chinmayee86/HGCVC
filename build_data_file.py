import mediapipe as mp
import cv2
import numpy as np
import uuid # universally unique identifier (to generate unique random string for image names)
import os
import copy
import itertools
import csv

# drawing utilities (makes it easy to render different landmarks on hand), landmark: x, y, and z coordinates of points  
mp_drawing = mp.solutions.drawing_utils

# mediapipe hands model
mp_hands = mp.solutions.hands

def calc_lmpoint_rect(image, landmarks):
    
    lmpoint = []
    
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        
        lmpoint.append([landmark_x, landmark_y])

        landmark_point = [np.array((landmark_x, landmark_y))]
        
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)
    
    rect = [x, y, x + w, y + h]

    return lmpoint, rect

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def logging_csv(number, landmark_list):
    if 0 <= number <= 24:
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

# take feed from webcam, pass it to mediapipe, make detections, render the results to the image
# to get webcam feed with real-time detections applied

with open('model/keypoint_classifier/keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

cap = cv2.VideoCapture(0) # webcam feed


# set detection and tracking confidence of mediapipe hand detection model 
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1, static_image_mode=True) as hands: 
    while cap.isOpened(): # if webcam feed is there
        
        key = cv2.waitKey(10)        
        number = -1
        if 97 <= key <= 121:  # a ~ y
            number = key - 97
        
        ret, frame = cap.read() # frame represent image from webcam
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # convert BGR format to RGB format to use with mediapipe
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False # make the information read only
        
        # Detections
        results = hands.process(image) # image processed by mediapipe to generate results 
        
        # Set flag to true
        image.flags.writeable = True # render onto the image
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # convert RGB to BGR format
        
        # Detections
        #print(results)q
        
        # Rendering results (superimpose mediapipe landmarks on webcam feed)
        if results.multi_hand_landmarks: # checking if any rendering results
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS, # mp_hands.HAND_CONNECTIONS - which landmark is connected to what
                                        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4), # BGR 
                                        mp_drawing.DrawingSpec(color=(224, 224, 224), thickness=2, circle_radius=2),
                                         )
                
                lmpoint, rect = calc_lmpoint_rect(image, hand);
                
                
                
                pre_processed_landmark_list = pre_process_landmark(lmpoint)
                
#                 print(pre_processed_landmark_list)
                
                logging_csv(number, pre_processed_landmark_list)

        cv2.imshow('Hand Tracking', image)
        
        

        if cv2.waitKey(10) & 0xFF == ord('z'): 
            break

cap.release() # 
cv2.destroyAllWindows() # closes down frame

