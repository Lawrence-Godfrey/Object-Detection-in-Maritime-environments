import cv2
import numpy as np
import sys
import argparse
import os
import json 

import tensorflow as tf
import segmentation_models as sm

parser = argparse.ArgumentParser(description='Train a segmentation model on a video dataset')

parser.add_argument('--input_json_file', type=str, metavar='', default=None, help='The path to the json file containing the dataset file names')

parser.add_argument('-i', '--input_Video_folders',      type=str, nargs='+', metavar='', default=None, help='The path to the folder containing the videos to train on')
parser.add_argument('-m', '--input_Mask_folders',       type=str, nargs='+', metavar='', default=None, help='The path to the folder containing the masked videos to train on')

parser.add_argument('-c', '--Model_Checkpoint_Folder', type=str, metavar='',   help='The folder to save model checkpoints')
parser.add_argument('-b', '--Model_type',              type=str, metavar='',   default='vgg19', help='The model backbone. Default=vgg19')
parser.add_argument('-s', '--show_video', 			   action='store_true', metavar='', help='Whether or not to show the input and mask video while reading it in')
parser.add_argument('-p', '--percent_val',             type=float, metavar='', default = 0.2,    help='percentage of dataset to use as validation')

args = parser.parse_args()

if args.input_Video_folder is not None:
	input_folders = args.input_Video_folder
	input_mask_folders = args.input_Mask_folder
	
	all_input_filenames, all_mask_filenames = [],[]

	for input_folder, input_mask_folder in zip(input_folders, input_mask_folders):
		all_input_filenames.append((input_folder + filename for filename in os.listdir(input_folder)))
		all_mask_filenames.append((input_mask_folder + filename for filename in os.listdir(input_mask_folder)))

	print(all_input_filenames)
	print(all_mask_filenames)

elif args.input_json_file is not None:
	json_file = args.input_json_file
	
	with open(json_file) as file:
		data = json.load(file)

	all_input_filenames = data['Videos']['SMD']
	all_mask_filenames = data['Masks']['SMD']

else:
	sys.exit("Either --input_json_file or --input_Video_folder required")

sys.exit()
checkpoint_path = args.Model_Checkpoint_Folder


# Set up model
num_classes = 1

BACKBONE = args.Model_type
preprocess_input = sm.get_preprocessing(BACKBONE)

model = sm.Unet(BACKBONE, encoder_weights='imagenet', classes=num_classes)

model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.iou_score],
)

# model.summary()

all_frames, all_masks = [], []

for filename, mask_filename in zip(all_input_filenames, all_mask_filenames):
	print('Reading ' + filename)


	# read in files 
	vid = cv2.VideoCapture(filename)
	mask_vid = cv2.VideoCapture(mask_filename)

	if not vid.isOpened():
		sys.exit('Video File, ' + filename + ', couldn\'t be opened')
	
	if not mask_vid.isOpened():
		sys.exit('Video File, ' + mask_filename + ', couldn\'t be opened')

	xSize = 320
	ySize = 320

	while vid.isOpened() and mask_vid.isOpened:

		available, frame = vid.read()
		_, mask_frame = mask_vid.read()

		if available:

			frame = cv2.resize(frame, (xSize, ySize), interpolation=cv2.INTER_AREA)
			
			if args.show_video:
				cv2.imshow('Video', frame)

			mask_frame = cv2.resize(mask_frame, (xSize, ySize), interpolation=cv2.INTER_AREA)
			mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
			
			# convert single array into 4 arrays. This is necessary for training
			label = np.where(mask_frame>240, 1, 0).astype(np.uint8)
			label = label.reshape((320, 320, 1))

			if args.show_video:
				cv2.imshow('Mask', mask_frame)

			all_frames.append(frame)
			all_masks.append(label)

			if args.show_video:
				if cv2.waitKey(1) == 27:
					vid.release()
					mask_vid.release()
					cv2.destroyAllWindows()
					sys.exit('Video closed')

		else:
			print('Frame not available')
			break

	vid.release()
	mask_vid.release()
	if args.show_video:
		cv2.destroyAllWindows()


# calculate number of frames to use for training
num_train_examples = int(len(all_frames)*(1-args.percent_val))

# convert lists to numpy arrays
all_frames, all_masks = np.array(all_frames), np.array(all_masks)

# Shuffle the arrays
rng_state = np.random.get_state()
np.random.shuffle(all_frames)
np.random.set_state(rng_state)
np.random.shuffle(all_masks)

# split whole video into training and validation
x_train, y_train,  = all_frames[0:num_train_examples], all_masks[0:num_train_examples]
x_val,   y_val    =  all_frames[num_train_examples:],  all_masks[num_train_examples:]

# reshape to make sure it's in the right format for tensorflow
# x_train, y_train, x_val, y_val = x_train.reshape((-1, xSize, ySize, 3)), y_train.reshape((-1, xSize, ySize, num_classes)), x_val.reshape((-1, xSize, ySize, 3)), y_val.reshape((-1, xSize, ySize, num_classes))

# preprocess input
x_train = preprocess_input(x_train)
x_val   = preprocess_input(x_val)



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
	validation_steps=100,
    steps_per_epoch=5,
    epochs=50,
	callbacks=[cp_callback] 
)