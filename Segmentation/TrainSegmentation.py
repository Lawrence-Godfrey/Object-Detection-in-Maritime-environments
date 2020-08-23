import cv2
import numpy as np
import sys
import argparse
import os

import tensorflow as tf
import segmentation_models as sm

parser = argparse.ArgumentParser(description='Train a segmentation model on a video dataset')
parser.add_argument('-i', '--Input_Video_folder',      type=str, metavar='', help='The path to the folder containing the videos to train on')
parser.add_argument('-m', '--Input_Mask_folder',       type=str, metavar='', help='The path to the folder containing the masked videos to train on')
parser.add_argument('-c', '--Model_Checkpoint_Folder', type=str, metavar='', help='The folder to save model checkpoints')
parser.add_argument('-b', '--Model_type',              type=str, metavar='', default='vgg19', help='The model backbone. Default=vgg19')
args = parser.parse_args()


input_folder = args.Input_Video_folder
input_mask_folder = args.Input_Mask_folder

checkpoint_path = args.Model_Checkpoint_Folder

# Set up model
BACKBONE = args.Model_type
preprocess_input = sm.get_preprocessing(BACKBONE)

model = sm.Unet(BACKBONE, encoder_weights='imagenet')

model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

model.summary()

for filename in os.listdir(input_folder):

	all_frames, all_masks = [], []

	# read in files 
	vid = cv2.VideoCapture(input_folder + filename)
	mask_vid = cv2.VideoCapture(input_mask_folder + filename)

	if not vid.isOpened():
		sys.exit('Video File, ' + input_folder + filename + ', couldn\'t be opened')
	
	if not mask_vid.isOpened():
		sys.exit('Video File, ' + input_mask_folder + filename + ', couldn\'t be opened')

	xSize = 320
	ySize = 320

	i=0
	while vid.isOpened() and mask_vid.isOpened:

		available, frame = vid.read()
		_, mask_frame = mask_vid.read()

		if available:

			frame = cv2.resize(frame, (xSize, ySize), interpolation=cv2.INTER_AREA)
			# cv2.imshow('Video', frame)

			mask_frame = cv2.resize(mask_frame, (xSize, ySize), interpolation=cv2.INTER_AREA)
			mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
			# cv2.imshow('Video2', mask_frame)


			all_frames.append(frame)
			all_masks.append(mask_frame)


			i+=1
			if cv2.waitKey(1) == 27:
				break

		else:
			print('Frame not available')
			break

	vid.release()
	mask_vid.release()
	cv2.destroyAllWindows()


	x_train, y_train, x_val, y_val = np.array(all_frames[0:200]), np.array(all_masks[0:200]), np.array(all_frames[200:273]), np.array(all_masks[200:273])

	x_train, y_train, x_val, y_val = x_train.reshape((200, xSize, ySize, 3)), y_train.reshape((200, xSize, ySize, 1))/255, x_val.reshape((73, xSize, ySize, 3)), y_val.reshape((73, xSize, ySize, 1))/255

	# preprocess input
	x_train = preprocess_input(x_train)
	x_val   = preprocess_input(x_val)

	# train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
	# test_data = tf.data.Dataset.from_tensor_slices((x_val, y_val))

	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
													save_weights_only=True,
													verbose=1)


	model.fit(
		x=x_train,
		y=y_train,
		batch_size=16,
		epochs=1,
		validation_data=(x_val, y_val),
		callbacks=[cp_callback]  # Pass callback to training
	)