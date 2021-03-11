# DJI Tello Visual Gesture control

## Contains of the repo

## Index
1. [Introduction](#Introduction)
2. [Setup](#Setup)
    1. [Install pip packages](#1.-Installing-pip-packages)
    2. [Connect and test Tello](#2.-Connect-Tello)
3. [Usage](#Usage)
    1. [Keyboard control](#Keyboard control)
    2. [Gesture control](#Gesture control)
4. [Adding new gestures](#Adding new gestures)
    1. [Technical description](#Technical details of gesture detector)
    2. [Creating dataset](#Creating dataset with new geastures)
    3. [Retrain model](#Notebook for retraining model)
5. [Repository structure](#Repository structure)
    
## Introduction
...
## Setup
### 1. Installing pip packages
First, we need to install python dependencies. Make sure you that you are using `python3.7`

List of packages
```sh
ConfigArgParse == 1.2.3
djitellopy == 1.5
numpy == 1.19.3
opencv_python == 4.5.1.48
tensorflow == 2.4.1
mediapipe == 0.8.2
```

Install
```sh
# requirements.txt
pip3 install -r requirements.txt
```
### 2. Connect Tello
Turn on drone and connect computer to its WiFi

Next, run the following code to verify connectivity

wifi.png

```sh
python3 tests/test_connection.py
```

On successful connection

```json
Send command: command
Response: b'ok'
```

If you get such output, you may need to check your connection with the drone

```json
Send command: command
Timeout exceed on command command
Command command was unsuccessful. Message: False
```

## Usage
The most interesting part is demo. There are 2 types of control: keyboard and gesture. You can change between control types during the flight. Below is a complete description of both types.

Run the following command to start the tello control :

```sh
python3 main.py
```

This script will start the python window with visualization like this:

WINDOW.img

### Keyboard control
(To control the drone with your keyboard, first press the `Left Shift` key.)

The following is a list of keys and action description -

* `Left Shift` -> Toggle Keyboard controls
* `Right Shft` -> Take off drone
* `Space` -> Land drone
* `Up arrow` -> Increase Altitude
* `Down arrow` -> Decrease Altitude
* `Left arrow` -> Pan left
* `Right arrow` -> Pan right
* `w` -> Move forward
* `a` -> Move left
* `s` -> Move down
* `d` -> Move right

### Gesture control 

By pressing `g` you activate gesture control mode. Here is a full list of gestures that are available now.

GESTURES_IMAGE.img

## Adding new gestures
Hand recognition detector can add and change training data to retrain the model on the own gestures. But before this,
there are technical details of the detector to understand how it works and how it can be improved
### Technical details of gesture detector
Mediapipe Hand keypoints recognition is returning 3D coordinated of 20 hand landmarks. For our
model we will use only 2D coordinates.

landmarks.png

Than, this points are preprocessed for training the model in the following way.

preprocessing.png

After that, we can use data to train our model. Keypoint calssifier is a simple Neural network with such 
structure

neural_network_structure.png 

*you can check how the structure was formed 
### Creating dataset with new gestures
Press "k" to enter the mode to save key points（displayed as 「MODE:Logging Key Point」）

mode.img

If you press "0" to "9", the key points will be added to "model/keypoint_classifier/keypoint.csv" as shown below.
1st column: Pressed number (used as class ID), 2nd and subsequent columns: Key point coordinates

table.img

In the initial state, 7 types of learning data are included as was shown [here](#Gesture control). If necessary, add 3 or later, or delete the existing data of csv to prepare the training data.
### Notebook for retraining model
Open "[Keypoint_model_training.ipynb](Keypoint_model_training.ipynb)" in Jupyter Notebook or Google Colab.
Change the number of training data classes,the value of "NUM_CLASSES = 3", and path to teh dataset. Then, execute all cells
and download `.tflite` model

showgif.gif

Do not forget to modify or add labels in `"model/keypoint_classifier/keypoint_classifier_label.csv"`

Bonus

The last part of the notebook is a grid search for model using TensorBoard. Run the GridSearch part of the notebook to
get test result with different parameters

grid_search.img

## Repository structure

# TODO
- [ ] Motion gesture support (LSTM)
- [ ] Web UI for mobile on-device gesture control
- [ ] Add [Holistic model](https://google.github.io/mediapipe/solutions/holistic) support

# Reference
* [MediaPipe](https://github.com/google/mediapipe)
* [MediaPipe Hand gesture recognition (by Kazuhito00)](https://github.com/Kazuhito00/hand-gesture-recognition-using-mediapipe)
* [Tello SDK python interface](https://github.com/damiafuentes/DJITelloPy)

# Author
Nikita Kiselov(https://github.com/kinivi)
 
# License 
tello-gesture-control is under [Apache-2.0 License](LICENSE).
