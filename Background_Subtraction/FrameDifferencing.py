import cv2
import numpy as np

vid = cv2.VideoCapture('Data/SingaporeMaritimeDataset/VIS_Onshore/Videos/MVI_1448_VIS_Haze.avi')

if not vid.isOpened():
	print('Video File couldn\'t be opened')

fps = int(vid.get(cv2.CAP_PROP_FPS))

_, frame1 = vid.read()
_, frame2 = vid.read()

cv2.namedWindow('Video Difference Blurred')
cv2.namedWindow('Video Difference Threshold')
cv2.namedWindow('Contours')

num_iterations=1
def setDilations(x):
	global num_iterations
	num_iterations=x

threshold_value=20
def setThreshold(x):
	global threshold_value
	threshold_value=x

isBlurred=False
def setBlurred(x):
	global isBlurred
	if x:
		isBlurred=True
	else:
		isBlurred=False


cv2.createTrackbar('Dilation Iterations', 'Contours', 0, 10, setDilations)
cv2.setTrackbarPos('Dilation Iterations', 'Contours', 3)

cv2.createTrackbar('Threshold Value', 'Contours', 0, 255, setThreshold)
cv2.setTrackbarPos('Threshold Value', 'Contours', 20)

cv2.createTrackbar('Is Blurred', 'Contours', 0, 1, setBlurred)

while vid.isOpened():

	difference = cv2.absdiff(frame1, frame2)
	difference = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)

	blurred = difference
	if isBlurred:
		blurred = cv2.GaussianBlur(difference, (5,5), sigmaX=0)

	_, threshold = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)
	
	dilated = cv2.dilate(threshold, kernel=None, iterations=num_iterations)
	
	contours, _ = cv2.findContours(dilated, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
	cv2.drawContours(frame1, contours, contourIdx=-1, color=(0,0,255), thickness=1)

	
	# cv2.imshow('Video Difference', difference)
	cv2.imshow('Video Difference Blurred', blurred)
	cv2.imshow('Video Difference Threshold', threshold)
	# cv2.imshow('Video Difference Threshold Dilated', dilated)

	n=1
	for contour in contours:
		(x,y,w,h) = cv2.boundingRect(contour)

		if cv2.contourArea(contour) > 900:
			cv2.rectangle(frame1, (x, y), (x+w, y+h), (0,255,0), thickness=1)
			cv2.putText(frame1, 'Detection ' + str(n), org=(x,y), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.6, color=(0,0,255), thickness=2)
			n+=1

	cv2.imshow("Contours", frame1)
	# cv2.imshow("Boxes", frame1)

	frame1 = frame2
	available, frame2 = vid.read()
	
	if not available:
		break

	if cv2.waitKey(1000//fps) == 27:
		break

vid.release()
cv2.destroyAllWindows()