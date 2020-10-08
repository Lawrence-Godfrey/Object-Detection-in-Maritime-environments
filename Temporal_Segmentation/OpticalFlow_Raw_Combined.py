import cv2
import numpy as np

import sys
import argparse
import os
from tqdm import tqdm 

import tensorflow as tf
import segmentation_models as sm

parser = argparse.ArgumentParser(description='Train a segmentation model on a video dataset')

parser.add_argument('-i', '--input_video_folders',      type=str, metavar='', nargs='+', required=True, help='The path to the folder containing the videos to train on')
parser.add_argument('-m', '--input_mask_folders',       type=str, metavar='', nargs='+', required=True, help='The path to the folder containing the masked videos to train on')

parser.add_argument('-t', '--input_test_video_folders', type=str, metavar='', nargs='+', required=True, help='The path to the folder containing the videos to test on')
parser.add_argument('-y', '--input_test_mask_folders',  type=str, metavar='', nargs='+', required=True, help='The path to the folder containing the masked videos to test on')

parser.add_argument('-b', '--model_type',              type=str, metavar='',   default='resnet18', help='The model backbone')
parser.add_argument('-c', '--model_checkpoint_folder', type=str, metavar='',   default = "./checkpoints/",    help='The folder to save model checkpoints')

parser.add_argument('-s', '--show_video', 			   action='store_true', help='Whether or not to show the input and mask video while reading it in')
parser.add_argument('-e', '--num_epochs', 				type=int, metavar='', default=10, help="Number of ephocs to train for")

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

# Set up model
num_classes = 1

# number of frames to take as input to the model
frame_window = 7

# Optical Flow Parameters
pyr_scale = 0.5
levels = 3
winsize = 4
iterations = 1
poly_n = 5
poly_sigma = 2


BACKBONE = args.model_type
preprocess_input = sm.get_preprocessing(BACKBONE)

model = sm.Unet(BACKBONE, encoder_weights=None, input_shape = (None, None, frame_window), classes=num_classes)

model.compile(
    'Adam',
    loss=sm.losses.bce_jaccard_loss,
    metrics=[sm.metrics.IOUScore(), sm.metrics.FScore()],
)

def preprocess_greyscale(x):
	x = x.astype(np.float32)
	x /= 255.
	# subtract mean 
	x -= 0.4867
	# scale by standard deviation
	x /= 0.2153
	return x

# Video Properties
frame_size = 320
frame_rate = 7


def read_frames_to_list(filename, x, is_mask=False, convert_to_flow=False):
	print('Reading ' + filename)

	# read in file
	vid = cv2.VideoCapture(filename)

	if not vid.isOpened():
		sys.exit('Video File, ' + filename + ', couldn\'t be opened')

	# Get the videos frame per second value
	vid_fps = vid.get(cv2.CAP_PROP_FPS)
	# destination dimensions of videos
	xSize = frame_size
	ySize = frame_size

	# Progress Bar 
	length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
	pbar = tqdm(total=length)

	frame_counter = 0
	previous_frame = None
	while vid.isOpened():

		# skip a certain number of frames, essentially making the frame rate 2 fps
		if frame_counter > vid_fps//frame_rate:
			frame_counter = 0
		
			available, frame = vid.read()

			if available:

				frame = cv2.resize(frame, (xSize, ySize), interpolation=cv2.INTER_AREA)
				
				if args.show_video:
					cv2.imshow('Video', frame)

				frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				
				if is_mask:	
					# convert single array into 4 arrays. This is necessary for training
					# sky = np.where(frame<=50, 1, 0).astype(np.uint8)
					frame = np.where(frame>=220, 1, 0).astype(np.uint8)
					# boat = np.where(frame==198, 1, 0).astype(np.uint8)
					# other = np.where(np.logical_and(mask_frame>90 , mask_frame<100), 1, 0).astype(np.uint8)

					# stack 4 into one array
					# frame = np.dstack((sky, sea, boat))

				if convert_to_flow:
					previous_frame = cv2.resize(previous_frame, (xSize, ySize), interpolation=cv2.INTER_AREA)
					previous_frame = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
					flow = cv2.calcOpticalFlowFarneback(previous_frame, frame, None, pyr_scale, levels, winsize, iterations, poly_n, poly_sigma, flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
					
					mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
					frame = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX).astype(np.float32)


				x.append(frame)

				pbar.update(1)
			
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
			_, previous_frame = vid.read()
			frame_counter += 1 
			pbar.update(1)


	pbar.close()
	vid.release()
	if args.show_video:
		cv2.destroyAllWindows()


# Read all training videos into these arrays
training_frames, training_flow_images, training_masks = [], [], []
test_frames, test_flow_images, test_masks = [], [], []

for filename, mask_filename in zip(training_files, training_mask_files):
	read_frames_to_list(filename, training_frames)
	read_frames_to_list(mask_filename, training_masks, is_mask=True)
	read_frames_to_list(filename, training_flow_images, convert_to_flow=True)
	assert len(training_frames) == len(training_masks), filename + " length not equal to " + mask_filename + " length"

for filename, mask_filename in zip(test_files, test_mask_files):
	read_frames_to_list(filename, test_frames)
	read_frames_to_list(mask_filename, test_masks, is_mask=True)
	read_frames_to_list(filename, test_flow_images, convert_to_flow=True)
	assert len(test_frames) == len(test_masks), filename + " length not equal to " + mask_filename + " length"


half_window = int(np.floor(frame_window/2))

# convert lists to numpy arrays
x_train = np.array(training_frames)
x_val = np.array(test_frames)

y_train, y_val = [], []


# batches of 7 frames each, for training on temporal data
training_batches, testing_batches = [], []

for i in range(half_window, x_train.shape[0] - half_window - 1, half_window + 1):
	temp = np.zeros((frame_size, frame_size, frame_window), dtype=np.float32)
	for j in range(0, frame_window, 1):
		frame = x_train[i+j-half_window, :, :]
		flow = training_flow_images[i+j-half_window]
		combined = frame + flow
		combined = np.where(combined>255, 255, combined)

		temp[...,j] = combined.astype(np.float32)


	training_batches.append(temp)
	y_train.append(training_masks[i])


for i in range(half_window, x_val.shape[0] - half_window - 1, half_window + 1):
	temp = np.zeros((frame_size, frame_size, frame_window), dtype=np.float32)
	for j in range(0, frame_window, 1):
		frame = x_val[i+j-half_window, :, :]
		flow = test_flow_images[i+j-half_window]
		
		combined = frame + flow
		combined = np.where(combined>255, 255, combined)

		temp[...,j] = combined.astype(np.float32)

	testing_batches.append(temp)
	y_val.append(test_masks[i])

# convert lists to numpy arrays
x_train, x_val = np.array(training_batches), np.array(testing_batches)

y_train, y_val = np.array(y_train), np.array(y_val)
y_train, y_val = np.reshape(y_train, (*y_train.shape, num_classes)), np.reshape(y_val, (*y_val.shape, num_classes))


key = 0
for i in range(x_train.shape[0]):
	cv2.imshow('mask', y_train[i].astype(np.uint8)*255)
	for j in range(frame_window):
		cv2.imshow('flow', x_train[i,:,:,j])
		key = cv2.waitKey(500)
	if key == 27:
		cv2.destroyAllWindows()
		break

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


image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args, channel_shift_range=50)
mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args, channel_shift_range=0.00000000001)

seed = 1
train_X_generator = image_datagen.flow(
	x_train,
    seed=seed)


train_Y_generator = mask_datagen.flow(
	y_train, 
    seed=seed)

# Visualize the batches with augmentation
# for image, mask in zip(train_X_generator, train_Y_generator):

# 	for i in range(image.shape[0]):
# 		for j in range(frame_window):
# 			cv2.imshow('image', image[i,:,:,j].astype(np.uint8))
# 			cv2.imshow('mask', mask[i].astype(np.uint8)*255)

# 			key = cv2.waitKey(1000)

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
    seed=seed)

val_Y_generator = mask_datagen.flow(
	y_val, 
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
	callbacks=[cp_callback, history_callback] 
)
