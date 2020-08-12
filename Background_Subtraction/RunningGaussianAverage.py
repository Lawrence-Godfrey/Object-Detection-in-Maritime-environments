import cv2
import numpy as np


vid = cv2.VideoCapture('/home/lawrence/FYProject/Data/MarDCT/20070928_1425_c04-C.m4v')

available, frame = vid.read()

if not available:
	vid.release()
	exit

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# initialize the mean to the first frame
mean = np.copy(frame)

# initialize the variances to a high number
var = np.full_like(mean, 150)

learning_rate = 0.05
T = 4

i = 0
while True:
	if i==100:
		break

	available, frame = vid.read()

	if available:                                
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  

		# calculate a new mean and variance based on previous
		new_mean = (1-learning_rate) * mean + learning_rate * frame_gray       
		new_mean = new_mean.astype(np.uint8)

		new_var  = (1-learning_rate) * var + learning_rate * cv2.subtract(frame_gray, mean)**2

		value = cv2.absdiff(frame_gray, mean)
		value = value / np.sqrt(var)


		mean = np.where(value < T, new_mean, new_mean)
		var =  np.where(value < T, new_var, new_var)

		background = np.where(value < T, np.uint8([0]), np.uint8([255]))

		cv2.imshow('background', background.astype(np.uint8))

		cv2.imwrite('ImagesForGIF/RunningGaussian/'+str(i).rjust(4, '0')+'.jpg', background.astype(np.uint8))

		i+=1 
		
		if cv2.waitKey(5) & 0xFF == 27:
				break
                
vid.release()                
cv2.destroyAllWindows()