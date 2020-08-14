import cv2
import numpy as np
import imutils
import time
'''Horizon Detection using Covariance Matrix'''

vid = cv2.VideoCapture('/home/lawrence/FYProject/Data/SingaporeMaritimeDataset/VIS_Onboard/Videos/MVI_0794_VIS_OB.avi')

if not vid.isOpened():
	print('Video File couldn\'t be opened')
	exit


def line(m, b, x, y):
	return y - m*x - b

xSize = 320
ySize = 240

# instead of looping over the i, j pixel positions, create two arrays for the row and columns values, 
# so that calculating the line values can be vectorised
row_vals = np.linspace(0, ySize-1, ySize)
row_vals = np.array([row_vals,]*xSize)
row_vals = row_vals.transpose()

col_vals = np.linspace(0, xSize-1, xSize)
col_vals = np.array([col_vals,]*ySize)

i = 0
while vid.isOpened():
	if i == 150:
		break

	available, frame = vid.read()
	if available:
		full_res = frame.copy()

		frame = cv2.resize(frame, (xSize, ySize), interpolation=cv2.INTER_AREA)
		cv2.imshow('Video', frame)

		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		blurred = cv2.GaussianBlur(frame_gray, ksize=(5, 5), sigmaX=0)

		horizon_intercept = 0
		horizon_slope = 0

		max_cost = 0.0

		# values of slope and intercept of the line to loop over 
		# res can be decreased for speed increase, but the horizon will be less accurate
		res = 50
		slope = np.linspace(-1,1,res*2)
		intercept = np.linspace(50,blurred.shape[0]-1,res)

		for m in range(len(slope)):
			for b in range(len(intercept)):
				# Find the pixels below and above the line
				line_vals = row_vals - (col_vals*slope[m]) - intercept[b]
				line_vals = line_vals*(-1*intercept[b])

				sky = np.where(line_vals>0, blurred, -1)
				ground = np.where(line_vals<=0, blurred, -1)

				# unravel into a single array
				sky = sky.ravel()
				ground = ground.ravel()

				# delete values that aren't part of the predicted sky area
				sky = np.delete(sky, np.where(sky==-1))

				# delete values that aren't part of the predicted ground area
				ground = np.delete(ground, np.where(ground==-1))

				cost = 1 / (np.var(sky) + np.var(ground))		
				
				if cost > max_cost:
					max_cost = cost
					horizon_intercept = intercept[b]
					horizon_slope = slope[m]

		print(i)
		
		# calculate the (x1,y1) and (x2,y2) positions to plot the line
		horizon_x1 = 0
		horizon_x2 = full_res.shape[1] - 1
		horizon_y1 = horizon_slope*horizon_x1 + horizon_intercept*4.5  # move the intercept so that it corresponds to the higher resolution frame
		horizon_y2 = horizon_slope*horizon_x2 + horizon_intercept*4.5  # ie. 1080/240 = 4.5 
		horizon_y1 = int(horizon_y1)
		horizon_y2 = int(horizon_y2)

		# Draw the line onto the frame
		full_res = cv2.line(full_res, pt1 = (horizon_x1, horizon_y1), pt2 = (horizon_x2, horizon_y2), color = (0, 0, 255), thickness = 3)

		cv2.imshow('Horizon line', full_res)
		cv2.imwrite('/home/lawrence/FYProject/ImagesForGIF/HorizonDetection/variance/'+str(i).rjust(4, '0')+'.jpg', full_res)
		i+=1 

		if cv2.waitKey(10) == 27:
			break
	else:
		break


vid.release()
cv2.destroyAllWindows()