<h3 align="center">OpenCV Age Detection with DeepLearing</h3>
<h3 align="center">Python3 :blue_heart:</h3>

<p>The Model Detects Age in Two States:</p>
<p>1.Age Detection in static Images</p>
<p>2.Age Detection in real-time video streams</p>

<p></p>

<p>Typically, you’ll see age detection implemented as a two-stage process:</p>
    <p>Stage 1: Detect faces in the input image/video stream</p>
    <p>Stage 2: Extract the face Region of Interest and apply the age detector algorithm to predict the age of the person</p>

Types of Face Detectors are : 
Haar cascades, HOG + Linear SVM, Single Shot Detectors (SSDs), etc.

<p>Deep learning-based face detectors are the most robust and will give you the best accuracy, but require even more computational resources than both Haar cascades and HOG + Linear SVMs</p>

<h5>In this project u have two Detector models</h5>

<p>1. detect_age.py: Single image age prediction</p>
<p>2. detect_age_video.py: Age prediction in video streams</p>
