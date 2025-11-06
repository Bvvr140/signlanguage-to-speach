# Import necessary modules
from flask import Blueprint  # Flask Blueprint for creating modular code
import os  # Operating system interface for file and directory management
import cv2  # OpenCV for video capture and image processing

# Define the directory for storing the dataset
DATA_DIR = './data'

# Create the data directory if it does not exist
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Define the number of classes (signs) and the dataset size per class
number_of_classes = 33
dataset_size = 100

# Initialize video capture from the default camera (index 0)
cap = cv2.VideoCapture(0)

# Loop through each class to collect data
for j in range(number_of_classes):
    # Create a directory for each class if it does not exist
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print('Collecting data for class {}'.format(j))

    # Wait for user readiness to start data collection
    while True:
        ret, frame = cap.read()  # Capture a frame from the video feed
        # Display instructions on the video feed
        cv2.putText(frame, 'Ready? Press "Q" ! :)', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('frame', frame)  # Show the frame in a window

        # Break the loop when the user presses 'q'
        if cv2.waitKey(25) == ord('q'):
            break

    # Collect 'dataset_size' number of images for the current class
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()  # Capture a frame from the video feed
        print("Captured frame shape:", frame.shape)  # Print frame shape for debugging
        cv2.imshow('frame', frame)  # Show the frame in a window
        cv2.waitKey(25)  # Wait for 25 milliseconds between frames
        # Save the captured frame to the corresponding class directory
        cv2.imwrite(os.path.join(class_dir, '{}.jpg'.format(counter)), frame)
        counter += 1  # Increment the counter

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
