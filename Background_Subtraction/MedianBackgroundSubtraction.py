import numpy as np
import cv2

vid = cv2.VideoCapture('/home/lawrence/FYProject/Data/MarDCT/20070928_1425_c04-C.m4v')

fps = int(vid.get(cv2.CAP_PROP_FPS))

prev_images = []         # Holds the N previous frames
N = 50					 # The number of frames to average over
threshold_value = 75     

i=0
while True:
	if i == 100:
		break

	available, frame = vid.read()

	if available:
		cv2.imshow('Video',frame)
		cv2.imwrite('/home/lawrence/FYProject/ImagesForGIF/MedianBS/Original/'+str(i).rjust(4, '0')+'.jpg', frame)

		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		prev_images.append(frame)

		# removing the images after every 50 image
		if len(prev_images)==N:
				prev_images.pop(0)

		background = np.array(prev_images)
		background = np.median(background, axis=0)
		background = background.astype(np.uint8)
		cv2.imshow('background', background)

		foreground = cv2.absdiff(frame, background)

		cv2.imshow('foreground',foreground)
		
		_, threshold = cv2.threshold(foreground, threshold_value, 255, cv2.THRESH_BINARY)
		
		cv2.imshow('Threshold', threshold)

		cv2.imwrite('/home/lawrence/FYProject/ImagesForGIF/MedianBS/Threshold/'+str(i).rjust(4, '0')+'.jpg', threshold)
		
		print(i)
		i+=1

		if cv2.waitKey(1000//fps) == 27:
			break


vid.release()

cv2.destroyAllWindows()	

# import cv2
# import numpy as np

# vid = cv2.VideoCapture('/home/lawrence/FYProject/Data/MarDCT/20070928_1425_c04-C.m4v')

# fps = int(vid.get(cv2.CAP_PROP_FPS))

# prev_images = []         # Holds the N previous frames
# N = 50					 # The number of frames to average over

# #taking 13 frames to estimate the background
# for i in range(13):
# 	available, frame = vid.read()

# 	if available:
# 		frame = cv2.flip(frame,1)
# 		frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
# 		prev_images.append(frame)

# # getting shape of the frame to create background
# row, col = frame.shape
# background = np.zeros([row,col],np.uint8)
# background = np.median(prev_images,axis=0)

# # median changes dtype, so revert to int
# background = background.astype(np.uint8)
# res = np.zeros([row,col],np.uint8)

# # converting interger 0 and 255 to type uint8
# a = np.uint8([255])
# b = np.uint8([0])

# # initialising i so that we can replace frames from images to get new frames
# i = 0

# # creating different kernels for erode and dilate openration. bigger for erode and smaller for dilate
# kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT,(2,4))
# kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(3,6))
# while vid.isOpened():

# 	available, frame = vid.read()

# 	if available:
# 		frame = cv2.flip(frame,1)
# 		frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

# 		prev_images[i%13] = frame1
# 		background = np.median(prev_images, axis=0)
# 		background = background.astype(np.uint8)

# 		# taking absolute difference otherwise having trouble in setting a particular value of threshold used in np.where
# 		res = cv2.absdiff(frame1,background)
# 		res = np.where(res>20, a, b)
# 		res = cv2.morphologyEx(res,cv2.MORPH_ERODE,kernel2)
# 		res = cv2.morphologyEx(res, cv2.MORPH_DILATE, kernel1)

# 		# to get the colored part of the generated mask res
# 		res = cv2.bitwise_and(frame,frame,mask=res)
# 		cv2.imshow('median',res)
# 		cv2.imshow('background',background)

# 		if cv2.waitKey(1) & 0xFF == 27:
# 			break

# 		i = i+1


# vid.release()
# cv2.destroyAllWindows()