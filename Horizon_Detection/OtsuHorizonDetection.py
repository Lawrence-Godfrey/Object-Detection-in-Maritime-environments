import cv2
import numpy as np

'''Horizon Detection using Otsus Thresholding Method'''

vid = cv2.VideoCapture('/home/lawrence/FYProject/Data/SingaporeMaritimeDataset/VIS_Onboard/Videos/MVI_0794_VIS_OB.avi')

if not vid.isOpened():
	print('Video File couldn\'t be opened')
	exit

i = 0
while vid.isOpened():
	if i == 150:
		break

	available, frame = vid.read()

	if available:

		cv2.imshow('Video', frame)

		cv2.imwrite('/home/lawrence/FYProject/ImagesForGIF/HorizonDetection/Original/'+str(i).rjust(4, '0')+'.jpg', frame)

		# Blurring helps the Otsu algorithm more easily find the threshold value to use
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(frame_gray, ksize=(5, 5), sigmaX=0)

		# Get the Otsu Threshold mask. 
		_, threshold = cv2.threshold(blurred, thresh=0, maxval=255, type=cv2.THRESH_OTSU) #cv2.THRESH_BINARY+ 

		# Invert so that the sea is white and sky black, not necessary but makes finding the line slightly easier.  
		threshold = np.invert(threshold)
		cv2.imshow('Threshold', threshold)
		
		# Closing, useful in removing small black patches in the white areas 
		closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel = np.ones((9,9), np.uint8))
		cv2.imshow('Closed', closed)

		# the points of the line are found: (x1, y1) is the point on the left of the horizon, (x2, y2) the point on the right
		horizon_x1 = 0
		horizon_x2 = frame_gray.shape[1] - 1
		horizon_y1 = max(np.where(closed[:, horizon_x1] == 0)[0])
		horizon_y2 = max(np.where(closed[:, horizon_x2] == 0)[0])

		# Draw the line onto the frame
		frame = cv2.line(frame, pt1 = (horizon_x1, horizon_y1), pt2 = (horizon_x2, horizon_y2), color = (0, 0, 255), thickness = 5)
		cv2.imshow('Horizon line', frame)

		i+=1 


		if cv2.waitKey(10) == 27:
			break
	else:
		break


vid.release()
cv2.destroyAllWindows()