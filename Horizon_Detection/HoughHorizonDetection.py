import cv2
import numpy as np
import sys

'''Horizon Detection using Hough Transform'''

vid = cv2.VideoCapture('/home/lawrence/FYProject/Data/SingaporeMaritimeDataset/VIS_Onboard/Videos/MVI_0794_VIS_OB.avi')

if not vid.isOpened():
	sys.exit('Video File couldn\'t be opened')

xSize = 1920
ySize = 1080

kernelSize = (8,8)
kSize = (5,5)
sigmaX = 10
CannyThreshold1 = 220
CannyThreshold2 = 250
CannyApertureSize = 5
ErosionIterations = 1
HoughThreshold = 100
HoughMinLineLength = 150
HoughMaxLineGap = 10

def setkernelSize(size):
	global kernelSize
	kernelSize = (size, size)

def setkSize(size):
	global kSize
	kSize = (5,5)

def setsigmaX(sigma):
	global sigmaX
	sigmaX = sigma

def setCannyThreshold1(threshold):
	global CannyThreshold1
	CannyThreshold1 = threshold

def setCannyThreshold2(threshold) :
	global CannyThreshold2
	CannyThreshold2 = threshold

def setCannyApertureSize(size):
	global CannyApertureSize
	if size%2==0:
		size -=1
	CannyApertureSize = size

def setErosionIteration(iters):
	global ErosionIterations
	ErosionIterations = iters

def setHoughThreshold(threshold):
	global HoughThreshold
	HoughThreshold = threshold

def setHoughMinLineLength(length):
	global HoughMinLineLength
	HoughMinLineLength = length

def setHoughMaxLineGap(gap):
	global HoughMaxLineGap
	HoughMaxLineGap = gap

cv2.namedWindow('With Lines')
cv2.createTrackbar('Set erosion Kernel Size', 'With Lines', 0, 20, setkernelSize)
cv2.createTrackbar('set blurring k Size', 'With Lines', 0, 20, setkSize)
cv2.createTrackbar('set blurring sigmaX', 'With Lines', 0, 20, setsigmaX)
cv2.createTrackbar('set Canny Threshold1', 'With Lines', 0, 250, setCannyThreshold1)
cv2.createTrackbar('set Canny Threshold2', 'With Lines', 0, 250, setCannyThreshold2)
cv2.createTrackbar('Set Canny Aperture Size', 'With Lines', 3, 7, setCannyApertureSize)
cv2.createTrackbar('Set Erosion Iterations', 'With Lines', 0, 20, setErosionIteration)
cv2.createTrackbar('Set Hough Threshold', 'With Lines', 0, 255, setHoughThreshold)
cv2.createTrackbar('Set Hough Min Line Length', 'With Lines', 0,255, setHoughMinLineLength)
cv2.createTrackbar('Set Hough Max Line Gap' , 'With Lines', 0,255, setHoughMaxLineGap)


i = 0
while vid.isOpened():
	if i == 150:
		break

	available, frame = vid.read()

	if available:
		frame = cv2.resize(frame, (xSize, ySize), interpolation=cv2.INTER_AREA)
		cv2.imshow('Video', frame)

		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		erosion = cv2.erode(frame_gray, kernel=np.ones(kernelSize,np.uint8), iterations = ErosionIterations)

		blurred = cv2.GaussianBlur(erosion, ksize=kSize, sigmaX=sigmaX)

		# Edge Detection (Canny Edge detection in this case)
		edges = cv2.Canny(blurred, threshold1=CannyThreshold1, threshold2=CannyThreshold2, apertureSize=CannyApertureSize)
		cv2.imshow('Edges', edges)
		
		cv2.imwrite('/home/lawrence/FYProject/ImagesForGIF/HorizonDetection/HoughTransformEdges/'+str(i).rjust(4, '0')+'.jpg', edges)

		# Hough Lines to get all lines from frame within certain thresholds
		lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=HoughThreshold, minLineLength=HoughMinLineLength, maxLineGap=HoughMaxLineGap)

		y1_max = 0
		y2_max = 0
		x1_max = 0
		x2_max = 0
		if lines is not None:
			for line in lines:
				x1, y1, x2, y2 = line[0]

				# calculate the length of the line and find the max length
				if (x2-x1)**2 + (y2-y1)**2 > (x2_max-x1_max)**2 + (y2_max-y1_max)**2:
					y1_max = y1
					y2_max = y2
					x1_max = x1
					x2_max = x2

			# we want the line to cross the whole frame,
			# this means calculating the new x1,y1 x2,y2 values
			# make the slope negative since the y1 and y2 are measured from the top of the frame downwards
			slope = -(y2_max-y1_max)/(x2_max-x1_max)

			# y = mx + b 
			y1_max = y1_max - slope * -x1_max
			y1_max = int(y1_max)
			
			y2_max = y2_max - slope * (xSize - x2_max)
			y2_max = int(y2_max)

			x1_max = 0
			x2_max = xSize

			# draw the line onto the frame
			cv2.line(frame, pt1=(x1_max,y1_max), pt2=(x2_max,y2_max), color=(0,0,255), thickness=2)

		cv2.imshow('With Lines', frame)

		cv2.imwrite('/home/lawrence/FYProject/ImagesForGIF/HorizonDetection/HoughTransform/'+str(i).rjust(4, '0')+'.jpg', frame)
		
		i+=1 


		if cv2.waitKey(10) == 27:
			break
	else:
		vid.release()
		vid.open('/home/lawrence/FYProject/Data/SingaporeMaritimeDataset/VIS_Onboard/Videos/MVI_0794_VIS_OB.avi')
		


vid.release()
cv2.destroyAllWindows()