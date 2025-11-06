# Import necessary modules
import os  # Operating system interface for file and directory management
import pickle  # Module for serializing and deserializing Python objects
import mediapipe as mp  # MediaPipe for hand detection and landmark processing
import cv2  # OpenCV for image processing
import matplotlib.pyplot as plt  # Matplotlib for plotting (not used in this script)

# Initialize MediaPipe's hand detection and drawing utilities
mp_hands = mp.solutions.hands  # Hands solution from MediaPipe
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities for visualization
mp_drawing_styles = mp.solutions.drawing_styles  # Predefined drawing styles for landmarks

# Configure MediaPipe Hands for static image processing
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define the directory where the dataset is stored
DATA_DIR = './data'

# Initialize lists to store data and labels
data = []  # List to store processed landmark data
labels = []  # List to store corresponding labels for each data entry

# Loop through each class directory in the dataset directory
for dir_ in os.listdir(DATA_DIR):
    # Loop through each image in the current class directory
    for img_path in os.listdir(os.path.join(DATA_DIR, dir_)):
        data_aux = []  # Auxiliary list to store normalized landmark coordinates for the current image
        x_ = []  # List to store x-coordinates of landmarks
        y_ = []  # List to store y-coordinates of landmarks

        # Read the image using OpenCV and convert it to RGB format
        img = cv2.imread(os.path.join(DATA_DIR, dir_, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Process the image to detect hand landmarks
        results = hands.process(img_rgb)
        
        # Check if any hand landmarks are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract x and y coordinates of each landmark
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                # Debugging prints to check lengths of coordinates
                print("Length of x_:", len(x_))
                print("Length of y_:", len(y_))

                # Normalize landmark coordinates and add to data_aux
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            # Add the processed data and label to their respective lists
            data.append(data_aux)
            labels.append(dir_)

# Serialize and save the processed data and labels using pickle
f = open('data.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, f)
f.close()
