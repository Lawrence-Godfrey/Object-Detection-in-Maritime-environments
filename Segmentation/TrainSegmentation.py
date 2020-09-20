import cv2
import numpy as np

import sys
import argparse
import os
import json
from tqdm import tqdm 

import tensorflow as tf
import segmentation_models as sm

#TODO Allow for test video folder input 
#TODO Save history as .pickle
#TODO More augmentation
#TODO Automatic end when plateau
#TODO Try with grayscale input
#TODO Save linux image

parser = argparse.ArgumentParser(description='Train a segmentation model on a video dataset')

parser.add_argument('-i', '--input_video_folders',      type=str, metavar='', nargs='+', required=True, help='The path to the folder containing the videos to train on')
parser.add_argument('-m', '--input_mask_folders',       type=str, metavar='', nargs='+', required=True, help='The path to the folder containing the masked videos to train on')

parser.add_argument('-t', '--input_test_video_folders', type=str, metavar='', nargs='+', required=True, help='The path to the folder containing the videos to test on')
parser.add_argument('-y', '--input_test_mask_folders',  type=str, metavar='', nargs='+', required=True, help='The path to the folder containing the masked videos to test on')

parser.add_argument('-b', '--model_type',              type=str, metavar='',   default='vgg19', help='The model backbone. Default=vgg19')
parser.add_argument('-c', '--model_checkpoint_folder', type=str, metavar='',   default = "./checkpoints/",    help='The folder to save model checkpoints')
parser.add_argument('-s', '--show_video', 			   action='store_true', help='Whether or not to show the input and mask video while reading it in')


args = parser.parse_args()

# read in filenames from arguments
training_folders = args.input_video_folders
training_mask_folders = args.input_mask_folders

test_folders = args.input_test_video_folders
test_mask_folders = args.input_test_mask_folders

training_files, training_mask_files = [],[]
test_files, test_mask_files = [],[]

for folder, mask_folder in zip(training_folders, training_mask_folders):
	for filename in os.listdir(folder):
		training_files.append(folder + filename) 

	for filename in os.listdir(mask_folder):
		training_mask_files.append(mask_folder + filename)


checkpoint_path = args.model_checkpoint_folder

# Set up model
num_classes = 1

BACKBONE = args.model_type
preprocess_input = sm.get_preprocessing(BACKBONE)

model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=num_classes)

model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

def read_frames_to_list(filename, x, is_mask=False):
	print('Reading ' + filename)

	# read in file
	vid = cv2.VideoCapture(filename)

	if not vid.isOpened():
		sys.exit('Video File, ' + filename + ', couldn\'t be opened')


	# destination dimensions of videos
	xSize = 320
	ySize = 320

	# Progress Bar 
	length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	pbar = tqdm(total=length)

	while vid.isOpened():

		available, frame = vid.read()

		if available:

			frame = cv2.resize(frame, (xSize, ySize), interpolation=cv2.INTER_AREA)
			
			if args.show_video:
				cv2.imshow('Video', frame)

			if is_mask:
				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				
				# convert single array into 4 arrays. This is necessary for training
				sky = np.where(frame<=50, 255, 0).astype(np.uint8)
				sea = np.where(frame>=220, 255, 0).astype(np.uint8)
				boat = np.where(frame==198, 255, 0).astype(np.uint8)
				# other = np.where(np.logical_and(mask_frame>90 , mask_frame<100), 1, 0).astype(np.uint8)

				# stack 4 into one array
				frame = np.dstack((sky, sea, boat))
				cv2.imshow('label', frame)

			x.append(frame)

			pbar.update(1)
		
			if args.show_video:
				if cv2.waitKey(30) == 27:
					vid.release()
					cv2.destroyAllWindows()
					pbar.close()
					sys.exit('Video closed')

		else:
			break

	pbar.close()
	vid.release()
	if args.show_video:
		cv2.destroyAllWindows()


# Read all training videos into these arrays
training_frames, training_masks = [], []
test_frames, test_masks = [], []

for filename, mask_filename in zip(training_files, training_mask_files):
	read_frames_to_list(filename, training_frames)
	read_frames_to_list(mask_filename, training_masks, is_mask=True)

	assert len(training_frames) == len(training_masks), filename + " length not equal to " + mask_filename + " length"

for filename, mask_filename in zip(test_files, test_mask_files):
	read_frames_to_list(filename, test_frames)
	read_frames_to_list(mask_filename, test_masks, is_mask=True)

	assert len(test_frames) == len(test_masks), filename + " length not equal to " + mask_filename + " length"

sys.exit()


# convert lists to numpy arrays
x_train, y_train = np.array(training_frames), np.array(training_masks)
x_val, y_val = np.array(test_frames), np.array(test_masks)

# preprocess input
x_train = preprocess_input(x_train)
x_val = preprocess_input(x_val)

# save checkpoints while training
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
												save_weights_only=True,
												verbose=1)



# create two datagen instances with the same arguments
data_gen_args = dict(   rotation_range=20,
						width_shift_range=0.2,
						height_shift_range=0.2,
						horizontal_flip=True)


image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

seed = 1
train_X_generator = image_datagen.flow(
	x_train,
    seed=seed)
	
train_Y_generator = mask_datagen.flow(
	y_train, 
    seed=seed)

val_X_generator = image_datagen.flow(
	x_val,
    seed=seed)

val_Y_generator = mask_datagen.flow(
	y_val, 
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(train_X_generator, train_Y_generator)
validation_generator = zip(val_X_generator, val_Y_generator)

model.fit(
    x=train_generator,
	validation_data=validation_generator,
	validation_steps=50,
    steps_per_epoch=1000,
    epochs=50,
	callbacks=[cp_callback] 
)