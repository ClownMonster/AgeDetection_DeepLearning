'''
Detecting the Age of a person from his static Images

'''
# imports
import cv2
import numpy as np
import os

# list of age buckets our age detector will predict
AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
	"(38-43)", "(48-53)", "(60-100)"]

# loading face detector model
print("[INFO] loading face detector model...")

prototxtPath = "./face_detector/deploy.prototxt"
weightsPath = "./face_detector/res10_300x300_ssd_iter_140000.caffemodel" 

faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# loading our serialized age detector model
print("[INFO] loading age detector model...")
prototxtPath = "./age_detector/age_deploy.prototxt"
weightsPath = "./age_detector/age_net.caffemodel"
ageNet = cv2.dnn.readNet(prototxtPath, weightsPath)



# loading the input image and constructing an input blob for the image
image = cv2.imread("./images/messi.jpg") # change path for another image
(h, w) = image.shape[:2]
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
(104.0, 177.0, 123.0))

# detecting the faces in the input image
print("[INFO] computing face detections...")
faceNet.setInput(blob)
detections = faceNet.forward()

# loopin over the detections(faces)
for i in range(0, detections.shape[2]):
	confidence = detections[0, 0, i, 2]
	# filtering out weak detections by ensuring the confidence is greater than the minimum confidence
	if confidence > 0.5:
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# extracting the Regin of Intrest of the face and then constructing a blob 
		face = image[startY:endY, startX:endX]
		faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
			(78.4263377603, 87.7689143744, 114.895847746),
			swapRB=False)

        # making predictions on the age and find the age bucket with
		# the largest corresponding probability
		ageNet.setInput(faceBlob)
		preds = ageNet.forward()
		i = preds[0].argmax()
		age = AGE_BUCKETS[i]
		ageConfidence = preds[0][i]

		# displaying the predicted age to our terminal
		text = "{}: {:.2f}%".format(age, ageConfidence * 100)
		print("[INFO] {}".format(text))
		
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(image, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(image, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

# displaying the output image
cv2.imshow("Image", image)
cv2.waitKey(0)