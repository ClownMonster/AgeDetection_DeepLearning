'''
Age Detection in video streams

'''
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_age(frame, faceNet, ageNet, minConf=0.5):
	AGE_BUCKETS = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)",
		"(38-43)", "(48-53)", "(60-100)"]

	# initializeing our results list
	results = []

	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	faceNet.setInput(blob)
	detections = faceNet.forward()

	# looping over the detections(faces in a frame)
	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2]

		if confidence > minConf:
	
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

		
			face = frame[startY:endY, startX:endX]

			if face.shape[0] < 20 or face.shape[1] < 20:
				continue

			faceBlob = cv2.dnn.blobFromImage(face, 1.0, (227, 227),
				(78.4263377603, 87.7689143744, 114.895847746),
				swapRB=False)

			ageNet.setInput(faceBlob)
			preds = ageNet.forward()
			i = preds[0].argmax()
			age = AGE_BUCKETS[i]
			ageConfidence = preds[0][i]

			d = {
				"loc": (startX, startY, endX, endY),
				"age": (age, ageConfidence)
			}
			results.append(d)

	return results

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


# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# looping over the frames from the video stream
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detecting faces in the frame, and for each face in the frame,
	# predicting the age
	results = detect_and_predict_age(frame, faceNet, ageNet,
		minConf=args["confidence"])

	# loop over the results
	for r in results:
		# drawing the bounding box of the face along with the associated predicted age
		text = "{}: {:.2f}%".format(r["age"][0], r["age"][1] * 100)
		(startX, startY, endX, endY) = r["loc"]
		y = startY - 10 if startY - 10 > 10 else startY + 10
		cv2.rectangle(frame, (startX, startY), (endX, endY),
			(0, 0, 255), 2)
		cv2.putText(frame, text, (startX, y),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

	# show the output frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
	if key == ord("q"):
		break
		
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()