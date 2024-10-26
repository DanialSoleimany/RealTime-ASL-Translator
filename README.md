# Real-Time American Sign Language Translator

## Overview
This project aims to assist the deaf and hard-of-hearing community in bridging communication gaps by leveraging computer vision and AI. The system enables real-time recognition of American Sign Language (ASL) gestures, providing an accessible way to interpret hand gestures into text. This tool could greatly enhance communication in various social, educational, and professional environments for people who rely on ASL to communicate.

The project collects data for ASL hand gestures through a webcam, extracts hand landmarks, and classifies gestures using machine learning. The model recognizes letters from "a" to "z," with adaptations for letters like "J" and "Z," which require hand movement. Additional gestures allow users to delete text, clear all text, and add spaces, offering a more comprehensive and user-friendly experience.

<p align="center">
  <b>Sign Language Alphabet (a-z)</b>
</p>

<p align="center">
  <img src="https://github.com/DanialSoleimany/RealTime-ASL-Translator/blob/main/American%20Sign%20Language.png" alt="Sign Language Alphabet">
</p>

<p align="center">––––––––––––––––––––––––––––––––––––––––––––</p>

<p align="center">
  <b>Demo Video</b>
</p>

<p align="center">
  <a href="https://www.youtube.com/watch?v=Qa3nPWC7lqM">
    <img src="https://img.youtube.com/vi/Qa3nPWC7lqM/0.jpg" alt="Watch the video" />
  </a>
</p>

## Table of Contents
1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Dataset](#dataset)
4. [Model](#model)
5. [Realtime Detection](#realtime-detection)
6. [Usage](#usage)
7. [License](#license)

## Requirements
To get started, ensure the following dependencies are installed:

- **Mediapipe**: `0.10.14`
- **OpenCV (cv2)**: `4.10.0`
- **Scikit-learn**: `1.5.2`

Install them using pip:

```bash
pip install mediapipe==0.10.14 opencv-python==4.10.0 scikit-learn==1.5.2
```

## Dataset
The dataset was self-collected with approximately 1000 images per ASL letter. Using the `collect_images.py` module, hand gestures were captured and labeled according to each letter. Since certain letters (e.g., "J" and "Z") require motion, static representations were chosen for these gestures to accommodate the model’s one-frame-at-a-time recognition limitation.

Using the `create_dataset.py` module, x and y coordinates of each hand landmark were extracted from the images and labeled for training. This structure helps achieve robustness against variations in hand positioning.

The project also includes custom gestures for:
- **Backspace**: Deletes the last word typed
- **Clear All**: Clears the entire text
- **Space**: Inserts a space between words
- **Unknown Gesture**: The model disregards any hand position not recognized as a valid ASL letter to minimize false predictions.

## Model: Random Forest Classifier for ASL Gesture Recognition

The model used in this project is a **Random Forest Classifier**, a robust and versatile algorithm that builds a collection (or "forest") of decision trees. This approach allows the classifier to make accurate predictions by combining results from multiple trees, thereby reducing the chances of overfitting and improving generalization. Below is a breakdown of how this model is applied to ASL gesture recognition:

- **Feature Extraction**:
  - The model operates on features extracted from hand landmarks. For each ASL gesture image, we use **MediaPipe** to identify the x and y coordinates of key hand landmarks.
  - These landmarks are normalized, making the model invariant to hand positioning and distance from the camera. This normalization helps ensure that the features accurately represent the gesture, regardless of slight variations in positioning.

- **Data Preparation**:
  - Each gesture is represented by a unique pattern of landmark coordinates. After extracting these coordinates, we label each set with the corresponding ASL letter.
  - Given ASL's requirement for high accuracy in gesture differentiation, we collected approximately 1000 images per letter to create a well-balanced dataset.

- **Model Training**:
  - Using the `train_classifier.py` module, we fed the extracted landmark coordinates and their labels into a Random Forest Classifier.
  - The data is split into training and testing sets to validate the model’s performance. During training, each tree in the forest learns to distinguish between different gesture patterns by making "decisions" based on the landmark positions.
  - The trained model achieved an accuracy of over 99%, indicating high reliability in recognizing static ASL gestures.

- **Prediction in Real-Time**:
  - Once trained, the model is saved as a `.p` file, which is loaded by the `application.py` module during real-time prediction.
  - During operation, each frame captured from the webcam is processed to extract landmarks. These landmarks are fed into the model, which predicts the corresponding ASL letter.
  - Special gestures (backspace, clear, space) and an "unknown" class (for gestures that don't match any ASL letter) are also handled by the model, allowing for smooth sentence construction and error management.

## Realtime Detection

The `realtime_detection.py` module manages the live detection and display of ASL gestures. It captures video frames from the webcam, processes each frame to extract hand landmarks, and uses the trained model to predict the corresponding ASL character.

- **Delay Adjustment with Counter**:
  - To improve accuracy and user experience, a delay is introduced to control the rate at which detected ASL gestures are converted to text. This delay is managed through a counter.
  - You can adjust the counter value to modify the speed of ASL-to-text conversion. Setting a lower counter value will make the conversion faster, displaying letters more quickly. A higher counter value, on the other hand, slows the process down, allowing more time for each gesture to be accurately recognized.
  - The current setting balances speed and accuracy, allowing users to display ASL characters (a-z) at a comfortable pace, making it possible to spell out names and sentences without rushing the gestures.

The module provides a smooth real-time experience by handling gestures with additional controls for special actions (backspace, clear, and space) and gracefully managing gestures that don’t match any ASL letter.

## Usage
To run the ASL recognition system:

1. Ensure your webcam is connected.
2. Run the `application.py` module, which initializes the webcam and predicts ASL gestures in real time.
3. Detected letters are displayed as text on the screen, allowing for live sentence construction.

The module also recognizes special gestures, letting users delete, clear, and add spaces to the typed text for smoother communication.

## License
This project is licensed under the MIT License. You are free to use and adapt this code in your projects.
