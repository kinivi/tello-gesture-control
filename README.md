# DJI Tello Hand Gesture control
-------

🏆 This project featured in [Official Google Dev Blog](https://developers.googleblog.com/2021/09/drone-control-via-gestures-using-mediapipe-hands.html)

-------
The main goal of this project is to control the drone using hand gestures without any gloves or additional equipment.
Just camera on the drone or your smartphone(soon), laptop and human hand.<br>

<img alt="demo_gif" src="https://user-images.githubusercontent.com/13486777/111168690-fb2e9280-85aa-11eb-894f-fe70633072fd.gif">


## Index
1. [Introduction](#Introduction)
2. [Setup](#Setup)
    1. [Install pip packages](#1.-Installing-pip-packages)
    2. [Connect and test Tello](#2.-Connect-Tello)
3. [Usage](#Usage)
    * [Keyboard control](##Keyboard-control)
    * [Gesture control](#Gesture-control)
4. [Adding new gestures](#Adding-new-gestures)
    * [Technical description](#Technical-details-of-gesture-detector)
    * [Creating dataset](#Creating-dataset-with-new-gestures)
    * [Retrain model](#Notebook-for-retraining-model)
5. [Repository structure](#Repository-structure)

## Introduction
This project relies on two main parts - DJI Tello drone and Mediapipe fast hand keypoints recognition.

DJI Tello is a perfect drone for any kind of programming experiments. It has a rich Python API (also Swift is available) which helps to almost fully control a drone, create drone swarms and utilise its camera for Computer vision.

Mediapipe is an amazing ML platform with many robust solutions like Face mesh, Hand Keypoints detection and Objectron. Moreover, their model can be used on the mobile platforms with on-device acceleration.

Here is a starter-pack that you need:

<img alt="starter_pack" width="80%" src="https://user-images.githubusercontent.com/13486777/111294166-b65e3680-8652-11eb-8225-c1fb1e5b867d.JPG">

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
pip3 install -r requirements.txt
```
### 2. Connect Tello
Turn on drone and connect computer to its WiFi

<img width="346" alt="wifi_connection" src="https://user-images.githubusercontent.com/13486777/110932822-a7b30f00-8334-11eb-9759-864c3dce652d.png">


Next, run the following code to verify connectivity

```sh
python3 tests/connection_test.py
```

On successful connection

```json
1. Connection test:
Send command: command
Response: b'ok'


2. Video stream test:
Send command: streamon
Response: b'ok'
```

If you get such output, you may need to check your connection with the drone

```json
1. Connection test:
Send command: command
Timeout exceed on command command
Command command was unsuccessful. Message: False


2. Video stream test:
Send command: streamon
Timeout exceed on command streamon
Command streamon was unsuccessful. Message: False
```

## Usage
The most interesting part is demo. There are 2 types of control: keyboard and gesture. You can change between control types during the flight. Below is a complete description of both types.

Run the following command to start the tello control :

```sh
python3 main.py
```

This script will start the python window with visualization like this:

<img width="60%" alt="window" src="https://user-images.githubusercontent.com/13486777/111294470-09d08480-8653-11eb-895d-a8ca9f6a288d.png">


### Keyboard control
To control the drone with your keyboard at any time - press the `k` key.

The following is a list of keys and action description -

* `k` -> Toggle Keyboard controls
* `g` -> Toggle Gesture controls
* `Space` -> Take off drone(if landed) **OR** Land drone(if in flight)
* `w` -> Move forward
* `s` -> Move back
* `a` -> Move left
* `d` -> Move right
* `e` -> Rotate clockwise
* `q` -> Rotate counter-clockwise
* `r` -> Move up
* `f` -> Move down
* `Esc` -> End program and land the drone 


### Gesture control 

By pressing `g` you activate gesture control mode. Here is a full list of gestures that are available now.

<img alt="gestures_list" width="80%" src="https://user-images.githubusercontent.com/13486777/110933057-f1035e80-8334-11eb-8458-988af973804e.JPG">

## Adding new gestures
Hand recognition detector can add and change training data to retrain the model on the own gestures. But before this,
there are technical details of the detector to understand how it works and how it can be improved
### Technical details of gesture detector
Mediapipe Hand keypoints recognition is returning 3D coordinated of 20 hand landmarks. For our
model we will use only 2D coordinates.

<img alt="gestures_list" width="80%" src="https://user-images.githubusercontent.com/13486777/110933339-49d2f700-8335-11eb-9588-5f68a2677ff0.png">


Then, these points are preprocessed for training the model in the following way.

<img alt="preprocessing" width="80%" src="https://user-images.githubusercontent.com/13486777/111294503-11902900-8653-11eb-9856-a50fe96e750e.png">


After that, we can use data to train our model. Keypoint classifier is a simple Neural network with such 
structure

<img alt="model_structure" width="80%" src="https://user-images.githubusercontent.com/13486777/112172879-c0a5a500-8bfd-11eb-85b3-34ccfa256ec3.jpg">



_check [here](#Grid-Search) to understand how the architecture was selected_
### Creating dataset with new gestures
First, pull datasets from Git LFS. [Here](https://github.com/git-lfs/git-lfs/wiki/Installation) is the instruction of how 
to install LFS. Then, run the command to pull default csv files
```sh
git lfs install
git lfs pull
```

After that, run `main.py` and press "n" to enter the mode to save key points
(displayed as **MODE:Logging Key Point**）

<img width="60%" alt="writing_mode" src="https://user-images.githubusercontent.com/13486777/111301228-a185a100-865a-11eb-8a3c-fa4d9ee96d6a.png">


If you press "0" to "9", the key points will be added to [model/keypoint_classifier/keypoint.csv](model/keypoint_classifier/keypoint.csv) as shown below.<br>
1st column: Pressed number (class ID), 2nd and subsequent columns: Keypoint coordinates

<img width="90%" alt="keypoints_table" src="https://user-images.githubusercontent.com/13486777/111295338-ec4fea80-8653-11eb-9bb3-4d27b519a14f.png">

In the initial state, 7 types of learning data are included as was shown [here](#Gesture-control). If necessary, add 3 or later, or delete the existing data of csv to prepare the training data.
### Notebook for retraining model
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kinivi/tello-gesture-control/blob/main/Keypoint_model_training.ipynb)

Open [Keypoint_model_training.ipynb](Keypoint_model_training.ipynb) in Jupyter Notebook or Google Colab.
Change the number of training data classes,the value of **NUM_CLASSES = 3**, and path to the dataset. Then, execute all cells
and download `.tflite` model

<img width="60%" alt="notebook_gif" src="https://user-images.githubusercontent.com/13486777/111295516-1ef9e300-8654-11eb-9f59-6f7a85b99076.gif">


Do not forget to modify or add labels in `"model/keypoint_classifier/keypoint_classifier_label.csv"`

#### Grid Search
❗️ Important ❗️ The last part of the notebook is an experimental part of the notebook which main functionality is to test hyperparameters of the model structure. In a nutshell: grid search using TensorBoard visualization. Feel free to use it for your experiments.


<img width="70%" alt="grid_search" src="https://user-images.githubusercontent.com/13486777/111295521-228d6a00-8654-11eb-937f-a15796a3024c.png">


## Repository structure
<pre>
│  main.py
│  Keypoint_model_training.ipynb
│  config.txt
│  requirements.txt
│  
├─model
│  └─keypoint_classifier
│      │  keypoint.csv
│      │  keypoint_classifier.hdf5
│      │  keypoint_classifier.py
│      │  keypoint_classifier.tflite
│      └─ keypoint_classifier_label.csv
│ 
├─gestures
│   │  gesture_recognition.py
│   │  tello_gesture_controller.py
│   └─ tello_keyboard_controller.py
│          
├─tests
│   └─connection_test.py
│ 
└─utils
    └─cvfpscalc.py
</pre>
### app.py
Main app which controls the functionality of drone control and gesture recognition<br>
App also includes mode to collect training data for adding new gestures.<br>

### keypoint_classification.ipynb
This is a model training script for hand sign recognition.

### model/keypoint_classifier
This directory stores files related to gesture recognition.<br>

* Training data(keypoint.csv)
* Trained model(keypoint_classifier.tflite)
* Label data(keypoint_classifier_label.csv)
* Inference module(keypoint_classifier.py)

### gestures/
This directory stores files related to drone controllers and gesture modules.<br>

* Keyboard controller (tello_keyboard_controller.py)
* Gesture controller(tello_keyboard_controller.py)
* Gesture recognition module(keypoint_classifier_label.csv)

### utils/cvfpscalc.py
Module for FPS measurement.

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
