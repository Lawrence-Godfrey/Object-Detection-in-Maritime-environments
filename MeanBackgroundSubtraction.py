import numpy as np
import cv2

vid = cv2.VideoCapture('/home/lawrence/FYProject/Data/MarDCT/20070928_1425_c04-C.m4v')

fps = int(vid.get(cv2.CAP_PROP_FPS))

prev_images = []         # Holds the N previous frames
N = 50					 # The number of frames to average over
threshold_value = 75     

num_iterations_dilate = 2

i=0
while True:
	if i == 100:
		break

	available, frame = vid.read()

	if available:
		cv2.imshow('Video',frame)
		cv2.imwrite('/home/lawrence/FYProject/ImagesForGIF/MeanBS/Original/'+str(i).rjust(4, '0')+'.jpg', frame)

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		prev_images.append(frame)

		# removing the images after every 50 image
		if len(prev_images)==N:
				prev_images.pop(0)

		background = np.array(prev_images)
		background = np.mean(background, axis=0)
		background = background.astype(np.uint8)
		cv2.imshow('background', background)

		foreground = cv2.absdiff(frame, background)

		cv2.imshow('foreground',foreground)
		
		_, threshold = cv2.threshold(foreground, threshold_value, 255, cv2.THRESH_BINARY)
		
		cv2.imshow('Threshold', threshold)

		cv2.imwrite('/home/lawrence/FYProject/ImagesForGIF/MeanBS/Threshold/'+str(i).rjust(4, '0')+'.jpg', threshold)
		
		print(i)
		i+=1

		if cv2.waitKey(1000//fps) == 27:
			break


vid.release()

cv2.destroyAllWindows()	