# Facial-Emotion-Recognition-using-Deep-Learning 
This repository implements a deep convolutional neural network (CNN) for facial expression recognition using the FER2013 dataset. The model classifies seven emotions: angry, disgust, fear, happy, neutral, sad, and surprise.Dataset E-link: https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset.

# Requirements

1. Python 3.11

2. Dependencies listed in requirements.txt

3. OpenCV Haar cascade file (haarcascade_frontalface_default.xml)
	
# Installation

1. Clone the repo:
	```bash
		git clone https://github.com/your-username/Face-Expression-Recognition.git
		cd Facial-Emotion-Recognition
	
2. Install dependencies:
	```bash
		pip install -r requirements.txt
3. Place FER2013 dataset in dataset/.
4. Download haarcascade_frontalface_default.xml to project root using https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml this link.

# Model

1. Architecture: 3 Conv2D blocks (64, 128, 256 filters), BatchNormalization, Dropout (0.25, 0.5), Dense (512 units, L2 regularization), softmax output.

2. Training: Adam optimizer, categorical crossentropy, early stopping, learning rate reduction.

# Files

1. main.py: Webcam emotion detection.

2. train_model.py: Model training/evaluation.

3. requirements.txt: Dependencies.

4. facial_emotion_model.h5 : Trained models.

5. haarcascade_frontalface_default.xml: Face detection cascade.


# Results
1. Test Accuracy: 0.6496%

2. Balanced Accuracy: 0.6515%

3. See confusion-matrix.png .




