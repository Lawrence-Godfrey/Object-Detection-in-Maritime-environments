import cv2
import numpy as np

# subtracts background using image with background only and image with object infront of background

vid = cv2.VideoCapture('Data/SingaporeMaritimeDataset/VIS_Onshore/Videos/MVI_1448_VIS_Haze.avi')

if not vid.isOpened():
	print('Video File couldn\'t be opened')

fps = int(vid.get(cv2.CAP_PROP_FPS))
print('delay =', 1000//fps)

bgsub_mog 			= cv2.bgsegm.createBackgroundSubtractorMOG()

bgsub_mog2 			= cv2.createBackgroundSubtractorMOG2()
bgsub_mog2_noshadow = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

bgsub_gmg 			= cv2.bgsegm.createBackgroundSubtractorGMG()
kernel 				= cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

bgsub_knn 			= cv2.createBackgroundSubtractorKNN()
bgsub_knn_noshadow  = cv2.createBackgroundSubtractorKNN(detectShadows=False)

i = 0
while vid.isOpened():
	available, frame = vid.read()

	if available:
		cv2.imshow('Video', frame)

		cv2.imwrite('ImagesForGIF/Original/'+str(i).rjust(4, '0')+'.jpg', frame)
		
		foreground_mask = bgsub_mog.apply(frame)
		cv2.imshow('MOG background subtraction', foreground_mask)
		
		foreground_mask = bgsub_mog2.apply(frame)
		cv2.imshow('MOG2 with shadows detected', foreground_mask)
		
		foreground_mask = bgsub_mog2_noshadow.apply(frame)
		cv2.imshow('MOG2 without shadows detected', foreground_mask)

		foreground_mask = bgsub_gmg.apply(frame)
		foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel=kernel)
		cv2.imshow('GMG background', foreground_mask)
		
		foreground_mask = bgsub_knn.apply(frame)
		cv2.imshow('KNN with shadows detected', foreground_mask)

		foreground_mask = bgsub_knn_noshadow.apply(frame)
		cv2.imshow('KNN without shadows detected', foreground_mask)

		cv2.imwrite('ImagesForGIF/KNN_BG_SUB/'+str(i).rjust(4, '0')+'.jpg', foreground_mask)

		i+=1

		if cv2.waitKey(10) == 27:
			break
	
	else:
		break

vid.release()
cv2.destroyAllWindows()