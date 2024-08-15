import mediapipe as mp
import cv2
import numpy as np
import uuid # universally unique identifier (to generate unique random string for image names)
import os
import copy
import itertools
import csv
from keras.models import load_model
import pickle
import time

# drawing utilities (makes it easy to render different landmarks on hand), landmark: x, y, and z coordinates of points  
mp_drawing = mp.solutions.drawing_utils

# mediapipe hands model
mp_hands = mp.solutions.hands

path_file = r"C:\Users\Harshita Rudraraju\Desktop\ML\Gesture_Recognition_Calculator\key_point_classification.sav"

loaded_model = pickle.load(open(path_file, 'rb'))

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

def draw_bounding_rect(image, brect):
    # Outer rectangle
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                 (0, 0, 0), 1)

    return image

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv2.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10: 
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11: 
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  
            cv2.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  
            cv2.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv2.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image

def draw_info_text(image, brect, label, hand_sign_text):
    
    cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)
    
    if hand_sign_text != "":
        info_text = label + ':' + hand_sign_text
    cv2.putText(image, info_text, (brect[0] + 5, brect[1] - 4),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return image

def get_result(first_number,operator,second_number):
    
    if str.isdigit(first_number) and str.isdigit(second_number):
        a = int(first_number)
        b = int(second_number)
        if operator == "*":
            res = a * b
        elif operator == "/":
            res = a/b
        elif operator == "+":
            res = a + b
        elif operator == "%":
            res = a % b
        elif operator == "-":
            res = a - b
        elif operator == "**":
            res = a**b
        else:
            print("Invalid operator")
            res = "NA"
        print("Result is {} {} {} = {}".format(first_number,operator,second_number,res))
    else:
        print("Invalid number entry")
        res = "NA"
    
    return res

# take feed from webcam, pass it to mediapipe, make detections, render the results to the image
# to get webcam feed with real-time detections applied

with open('keypoint_classifier_label.csv',
          encoding='utf-8-sig') as f:
    keypoint_classifier_labels = csv.reader(f)
    keypoint_classifier_labels = [
        row[0] for row in keypoint_classifier_labels
    ]

cap = cv2.VideoCapture(0) # webcam feed

fps = cap.get(cv2.CAP_PROP_FPS)

# print(fps)

first_number = ""
second_number = ""
operator = ""
sec = 0

# set detection and tracking confidence of mediapipe hand detection model 
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5, max_num_hands=1, static_image_mode=False) as hands: 
    while cap.isOpened(): # if webcam feed is there
                        
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
        
        # Rendering results (superimpose mediapipe landmarks on webcam feed)
        if results.multi_hand_landmarks: # checking if any rendering results
            for num, hand in enumerate(results.multi_hand_landmarks):
                
                sec = sec+1;
                
#                 print(sec)
                                
                lmpoint, rect = calc_lmpoint_rect(image, hand);
                
                pre_processed_landmark_list = pre_process_landmark(lmpoint)
                                                
                # Hand sign classification
                hand_sign_id = loaded_model.predict(np.array(pre_processed_landmark_list).reshape(1, -1))
                
                for i in results.multi_handedness:
                    label = i.classification[0].label
                                
                # Drawing part
                debug_image = draw_bounding_rect(image, rect)
                debug_image = draw_landmarks(image, lmpoint)
                debug_image = draw_info_text(image, rect, label, keypoint_classifier_labels[hand_sign_id[0]])
                
#                 # Open CV calculator
                
                if sec < 90: 
                
                    cv2.putText(image, 'Calculator Ready', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                
                elif 90 <= sec < 150:
                    
                    cv2.putText(image, 'Enter the first Number', (25, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3)
                    
                elif 150 <= sec < 420:
                    cv2.putText(image, first_number, (25, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3)
                    if (sec+150) % 90 == 0:
                        pred = keypoint_classifier_labels[hand_sign_id[0]]
                        if pred != "Confirm" and pred != "Clear":
                            first_number = first_number + pred
                        elif pred == "Clear":
                            sec = 150
                            first_number = ""
                        else:
                            sec = 420
                            
                elif 420 <= sec < 480:
                    cv2.putText(image, "Confirmed", (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3)
                    wor = "The first number is " + first_number
                    cv2.putText(image, wor, (25, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3)
                    
                                    
                elif 480 <= sec < 540:
                    cv2.putText(image, "Enter the operator", (25, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3)

                elif 540 <= sec < 630:
                    if (sec+60) % 60 == 0:
                        pred = keypoint_classifier_labels[hand_sign_id[0]]
                    if pred != "Confirm" and pred != "Clear":
                        operator = pred
                    elif pred == "Clear":
                        num = 540
                        operator = ""
                    else:
                        sec = 630
                
                elif 630 <= sec < 690:
                    cv2.putText(image, "Confirmed", (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3)
                    cv2.putText(image, operator, (25, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3)
                    
                    
                elif 690 <= sec < 750:
                    cv2.putText(image, 'Enter the Second Number', (25, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3)

                elif 750 <= sec < 1020:
                    cv2.putText(image, second_number, (25, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3)
                    if (sec+750) % 90 == 0:
                        pred = keypoint_classifier_labels[hand_sign_id[0]]
                        if pred != "Confirm" and pred != "Clear":
                            second_number = second_number + pred
                        elif pred == "Clear":
                            sec = 750
                            second_number = ""
                        else:
                            sec = 1020    
                                  
                elif 1020 <= sec < 1080:
                    cv2.putText(image, "Confirmed", (25, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3)
                    wor = "The second number is " + second_number
                    cv2.putText(image, wor, (25, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3)

                elif 1080 <= sec < 1350:
                    res = get_result(first_number,operator,second_number)
                    in_line = first_number + operator + second_number + " = " + str(res)
                    cv2.putText(image, "The answer is ", (25, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3)
                    cv2.putText(image,in_line,(25,275),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),3)
                
                elif 1350 <= sec < 1800:
                    cv2.putText(image, "Reset Calculator?", (25, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 3)
                    if (sec+1350) % 90 == 0:
                        pred = keypoint_classifier_labels[hand_sign_id[0]]
                        if pred == "Clear":
                            sec = 0
                            first_number = ""
                            operator = ""
                            second_number = ""
                
                elif 1800 <= sec:
                    cap.release()
                    cv2.destroyAllWindows()
                    
        else:
            sec = 0
            first_number = ""
            second_number = ""
            operator = ""
                    
        cv2.imshow('Hand Tracking', image)
                
        if cv2.waitKey(10) & 0xFF == ord('z'): 
            break

cap.release() # 
cv2.destroyAllWindows() # closes down frame