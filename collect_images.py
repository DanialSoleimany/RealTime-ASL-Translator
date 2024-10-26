import os 
import cv2

# Define the directory path where the dataset will be stored
dir_name = "c" # directory name that we want to use as the label of images -> ex. dir_name = f
data_dir = f"./ASL dataset/{dir_name}" # you can fill 'dir_name' to collect your own data

# Check if the directory already exists, and if not, create it
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

num_classes = 1
dataset_size = 1000

# Initialize video capture object to read from the default webcam (device 0)
cap = cv2.VideoCapture(0)

if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Display a "Ready" message until the user presses 'q' to proceed
while True:
    ret, frame = cap.read()  # Capture a frame from the webcam

    # Display a message on the frame
    cv2.putText(img=frame,                    # video frame to draw on
        text='Are you ready ?',               # text to display
        org=(100, 100),                       # bottom-left corner of the text (x, y) coordinates
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,    # font type
        fontScale=1,                          # size scale factor for the text
        color=(0, 255, 0),                    # text color in BGR format (Green here)
        thickness=3,                          # thickness of the text stroke
        lineType=cv2.LINE_AA)                 # Anti-aliased line type for smoother text
    
    # Show the frame with the message
    cv2.imshow("frame", frame)
    
    # Check if 'q' is pressed to quit the "Are you ready?" loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # Initialize counter to keep track of the number of images captured
    counter = 0
    # Collect 'dataset_size' number of images for the current class
    
while counter < dataset_size:
    ret, frame = cap.read()  # Capture the next frame

    # Display the current frame
    cv2.imshow("frame", frame)
    cv2.waitKey(1)

    # Save the frame as an image in the corresponding class folder
    cv2.imwrite(os.path.join(data_dir, f"{counter}.png"), frame)

    # Increment the counter after saving each image
    counter += 1

# Release the video capture object and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()