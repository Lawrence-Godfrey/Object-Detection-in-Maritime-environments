import cv2
import numpy as np
import sys

import tensorflow as tf
import segmentation_models as sm

filename = '/home/lawrence/FYProject/Data/MarDCT/20070928_1425_c04-C.m4v'

checkpoint_path = '/home/lawrence/FYProject/SavedModels/vgg19/'

# Set up model
BACKBONE = 'vgg19'
preprocess_input = sm.get_preprocessing(BACKBONE)

# load weights from training
model = sm.Unet(BACKBONE, encoder_weights='imagenet', weights=checkpoint_path)
model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score]
)


# read in video
vid = cv2.VideoCapture(filename)

if not vid.isOpened():
	sys.exit('Video File, ' + filename + ', couldn\'t be opened')

xSize = 320
ySize = 320

i=0
while vid.isOpened():
	if i == 274:
		print('finished')
		break

	available, frame = vid.read()

	if available:
		
		frame = cv2.resize(frame, (xSize, ySize), interpolation=cv2.INTER_AREA)
		
		cv2.imwrite('/home/lawrence/FYProject/ImagesForGIF/Segmentation/vgg19/original/'+str(i).rjust(4, '0')+'.jpg', frame)

		# prepare frame to be input to model
		frame = frame.reshape((1, xSize, ySize, 3))
		frame = preprocess_input(frame)

		cv2.imshow('Video', frame[0])
		
		# get the output of the model using frame as input
		prediction = model(frame, training=False)

		cv2.imwrite('/home/lawrence/FYProject/ImagesForGIF/Segmentation/vgg19/predicted/'+str(i).rjust(4, '0')+'.jpg', prediction.numpy()[0,:,:,0]*255)
		
		cv2.imshow('Prediction', prediction.numpy()[0,:,:,0]*255)


		i+=1
		if cv2.waitKey(1) == 27:
			break

	else:
		print('Frame not available')
		break

vid.release()
cv2.destroyAllWindows()
