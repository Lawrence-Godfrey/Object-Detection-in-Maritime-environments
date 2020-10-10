import cv2
import numpy as np
import sys

import tensorflow as tf
import segmentation_models as sm

filename = "/mnt/42F2C3CFF2C3C57F/Datasets/SingaporeMaritimeDataset/VIS_Onshore/Videos/MVI_1470_VIS.avi"

checkpoint_path = '/home/lawrence/FYProject/SavedModels/vgg19_4_classes/'

# Set up model
BACKBONE = 'vgg19'
preprocess_input = sm.get_preprocessing(BACKBONE)

# load weights from training
model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=4, weights=checkpoint_path)
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
	if i == 150:
		print('finished')
		break

	available, frame = vid.read()

	if available:
		
		frame = cv2.resize(frame, (xSize, ySize), interpolation=cv2.INTER_AREA)
		
		# cv2.imwrite('/home/lawrence/FYProject/ImagesForGIF/Segmentation/vgg19/original/'+str(i).rjust(4, '0')+'.jpg', frame)

		# prepare frame to be input to model
		frame = frame.reshape((1, xSize, ySize, 3))
		frame = preprocess_input(frame)

		cv2.imshow('Video', frame[0])
		
		# get the output of the model using frame as input
		prediction = model(frame, training=False)
		print(prediction.numpy().dtype)

		cv2.imshow('Prediction', (prediction.numpy()[0,:,:,0]*255).astype(np.uint8))

		# masked = np.zeros((xSize, ySize))
		# masked[np.where(prediction)]

		# cv2.imwrite('/home/lawrence/FYProject/ImagesForGIF/Segmentation/vgg19/predicted/'+str(i).rjust(4, '0')+'.jpg', prediction.numpy()[0,:,:,0]*255)
		
		# cv2.imshow('Prediction', prediction.numpy()[0,:,:,0]*255)


		i+=1
		if cv2.waitKey(100) == 27:
			break

	else:
		print('Frame not available')
		break

vid.release()
cv2.destroyAllWindows()
