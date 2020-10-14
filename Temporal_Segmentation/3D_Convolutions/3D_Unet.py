import tensorflow as tf
import numpy as np
import cv2

import sys
import argparse
import os
from tqdm import tqdm 

import models

import segmentation_models as sm

physical_devices = tf.config.list_physical_devices('GPU')

for gpu in physical_devices:
  tf.config.experimental.set_memory_growth(gpu, True)

parser = argparse.ArgumentParser(description='Train a segmentation model on a video dataset')

parser.add_argument('-i', '--input_video_folders',      type=str, metavar='', nargs='+', required=True, help='The path to the folder containing the videos to train on')
parser.add_argument('-m', '--input_mask_folders',       type=str, metavar='', nargs='+', required=True, help='The path to the folder containing the masked videos to train on')

parser.add_argument('-t', '--input_test_video_folders', type=str, metavar='', nargs='+', required=True, help='The path to the folder containing the videos to test on')
parser.add_argument('-y', '--input_test_mask_folders',  type=str, metavar='', nargs='+', required=True, help='The path to the folder containing the masked videos to test on')

parser.add_argument('-c', '--model_checkpoint_folder', type=str, metavar='',   default = "./checkpoints/",    help='The folder to save model checkpoints')

parser.add_argument('-s', '--show_video', 			   action='store_true', help='Whether or not to show the input and mask video while reading it in')
parser.add_argument('-e', '--num_epochs', 				type=int, metavar='', default=10, help="Number of ephocs to train for")
parser.add_argument('--batch_size', 				type=int, metavar='', default=32, help="Batch Size")
parser.add_argument('--frame_size', 				type=int, metavar='', default=320, help="Frame Size")

args = parser.parse_args()

# read in filenames from arguments
training_folders = args.input_video_folders
training_mask_folders = args.input_mask_folders

test_folders = args.input_test_video_folders
test_mask_folders = args.input_test_mask_folders

training_files, training_mask_files = [],[]
test_files, test_mask_files = [],[]

for folder, mask_folder in zip(training_folders, training_mask_folders):
	for filename in sorted(os.listdir(folder)):
		training_files.append(folder + filename) 

	for filename in sorted(os.listdir(mask_folder)):
		training_mask_files.append(mask_folder + filename)

for folder, mask_folder in zip(test_folders, test_mask_folders):
	for filename in sorted(os.listdir(folder)):
		test_files.append(folder + filename) 

	for filename in sorted(os.listdir(mask_folder)):
		test_mask_files.append(mask_folder + filename)

checkpoint_path = args.model_checkpoint_folder

batch_size = args.batch_size

num_classes = 1
frame_window = 4
frame_size = args.frame_size
input_shape = (frame_size, frame_size, frame_window, 1) # Width x Height x Depth x Channels

model = models.Unet3D(input_shape, n_filters=32, dropout=0.05)
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=sm.losses.bce_jaccard_loss, metrics=[sm.metrics.IOUScore(), sm.metrics.FScore()])
model.summary()

def preprocess_greyscale(x):
	x = x.astype(np.float32)
	x /= 255.
	# subtract mean 
	x -= 0.4829
	# scale by standard deviation
	x /= 0.2148
	return x


def read_frames_to_list(filename, x, is_mask=False):
	frame_rate = 10
	print('Reading ' + filename)

	# read in file
	vid = cv2.VideoCapture(filename)

	if not vid.isOpened():
		sys.exit('Video File, ' + filename + ', couldn\'t be opened')

	# Get the videos frame per second value
	vid_fps = vid.get(cv2.CAP_PROP_FPS)
	print(vid_fps)
	# destination dimensions of videos
	xSize = frame_size
	ySize = frame_size

	# Progress Bar 
	length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	pbar = tqdm(total=length//(vid_fps//frame_rate))
	print('total length before =', length)

	i=0
	frame_counter = 1
	while vid.isOpened():

		# skip a certain number of frames
		if frame_counter >= vid_fps//frame_rate:
			frame_counter = 1
		
			available, frame = vid.read()

			if available:
				# min_length = min(frame.shape[0], frame.shape[1])

				# frame = frame[:min_length, :min_length]

				frame = cv2.resize(frame, (xSize, ySize), interpolation=cv2.INTER_AREA)
				
				if args.show_video:
					cv2.imshow('Video', frame)

				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				
				if is_mask:	
					frame = np.where(frame>=220, 1, 0)


				x.append(frame)

				pbar.update(1)
				i+=1 

				if args.show_video:
					if cv2.waitKey(1) == 27:
						vid.release()
						cv2.destroyAllWindows()
						pbar.close()
						sys.exit('Video closed')

			else:
				break
		
		# skip frame using grab()
		else:
			_ = vid.grab()
			frame_counter += 1 

	print('total length =', i)
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


half_window = int(np.floor(frame_window/2))


x_train, x_val, y_train, y_val = [], [], [], []

for i in range(half_window, len(training_frames) - half_window, half_window + 1):
	image_batch = np.zeros((frame_size, frame_size, frame_window), dtype=np.float32)
	mask_batch = np.zeros((frame_size, frame_size, frame_window), dtype=np.float32)
	for j in range(0, frame_window, 1):
		image_batch[:,:,j] = training_frames[i+j-half_window]
		mask_batch[:,:,j] = training_masks[i+j-half_window]
		
	x_train.append(image_batch)
	y_train.append(mask_batch)

# set these to nothing to save clear memory
training_frames, training_masks = [], []

for i in range(half_window, len(test_frames) - half_window, half_window + 1):
	image_batch = np.zeros((frame_size, frame_size, frame_window), dtype=np.float32)
	mask_batch = np.zeros((frame_size, frame_size, frame_window), dtype=np.float32)
	for j in range(0, frame_window, 1):
		image_batch[:,:,j] = test_frames[i+j-half_window]
		mask_batch[:,:,j] = test_masks[i+j-half_window]

	x_val.append(image_batch)
	y_val.append(mask_batch)

# set these to nothing to save clear memory
test_frames, test_masks = [], []

# convert lists to numpy arrays
x_train = np.array(x_train)
x_val = np.array(x_val)
y_train = np.array(y_train)
y_val = np.array(y_val)

# key = 0
# for i in range(x_train.shape[0]):
# 	for j in range(frame_window):
# 		cv2.imshow('mask', y_train[i,:,:,j].astype(np.uint8)*255)
# 		cv2.imshow('input', x_train[i,:,:,j].astype(np.uint8))
# 		key = cv2.waitKey(500)
# 	if key == 27:
# 		cv2.destroyAllWindows()
# 		break


print(x_train.shape)
print(y_train.shape)
print(x_val.shape)
print(y_val.shape)

# preprocess input
x_train = preprocess_greyscale(x_train)
x_val = preprocess_greyscale(x_val)

# save checkpoints while training
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
												save_weights_only=False,
												save_best_only=True,
												monitor='val_f1-score',
												mode='max',
												verbose=1)
# Save history to csv file
history_callback = tf.keras.callbacks.CSVLogger(checkpoint_path + 'history.csv', append=True)

# early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.05, patience=3)


# create two datagen instances with the same arguments
data_gen_args = dict( 	width_shift_range=0.2,
						height_shift_range=0.2,
						horizontal_flip=True,
						fill_mode='reflect')


image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

seed = 1
train_X_generator = image_datagen.flow(
	x_train,
	batch_size=batch_size,
    seed=seed)


train_Y_generator = mask_datagen.flow(
	y_train, 
	batch_size=batch_size,
    seed=seed)

# Visualize the batches with augmentation
# for image, mask in zip(train_X_generator, train_Y_generator):
# 	print(image.shape)
# 	print(mask.shape)
# 	for i in range(image.shape[0]):
# 		for j in range(frame_window):
# 			cv2.imshow('image', (((image[i,:,:,j]*0.2148)+0.4829)*255).astype(np.uint8))
# 			cv2.imshow('mask', (mask[i,:,:,j]*255).astype(np.uint8))

# 			key = cv2.waitKey(500)

# 			if key == 27:
# 				cv2.destroyAllWindows()
# 				break
# 		if key == 27:
# 				cv2.destroyAllWindows()
# 				break
# 	if key == 27:
# 				cv2.destroyAllWindows()
# 				break

val_X_generator = image_datagen.flow(
	x_val,
	batch_size=batch_size,
    seed=seed)

val_Y_generator = mask_datagen.flow(
	y_val, 
	batch_size=batch_size,
    seed=seed)

# combine generators into one which yields image and masks
train_generator = zip(train_X_generator, train_Y_generator)
validation_generator = zip(val_X_generator, val_Y_generator)


history = model.fit(
	x=train_generator,
	validation_data=validation_generator,
	validation_steps=50,
	steps_per_epoch=300,
	epochs=args.num_epochs,
	callbacks=[cp_callback, history_callback], 
	max_queue_size=1
)