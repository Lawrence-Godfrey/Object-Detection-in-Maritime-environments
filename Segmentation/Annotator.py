import cv2
import numpy as np
import sys
import os
import argparse

parser = argparse.ArgumentParser(description='Annotate a video dataset using Otsu Thresholding')
parser.add_argument('-i', '--Input_Video_folder', type=str, metavar='',default='/home/lawrence/FYProject/Data/SingaporeMaritimeDataset/VIS_Onboard/Videos/', help='The path to the folder containing the videos')
parser.add_argument('-o', '--Output_Mask_folder', type=str, metavar='', default='/home/lawrence/FYProject/Data/SingaporeMaritimeDataset/VIS_Onboard/VideosAnnotated/', help='The path to the folder to output the masked videos')
parser.add_argument('-s', '--show_video', type=bool, metavar='', default=False, help='Whether or not to show the input and output video while converting')
args = parser.parse_args()

input_dir = args.Input_Video_folder
output_dir = args.Output_Mask_folder


for filename in os.listdir(input_dir):

	out_filename = output_dir + filename
	in_filename  = input_dir  + filename

	print('Converting ' + filename + ' ...')
	
	vid = cv2.VideoCapture(in_filename)

	if not vid.isOpened():
		sys.exit('Video File, ' + in_filename + ', couldn\'t be opened')

	# setup output video
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	resolution = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
	out = cv2.VideoWriter(filename = out_filename, fourcc = fourcc, fps = vid.get(cv2.CAP_PROP_FPS), frameSize = resolution, isColor = False)

	if not out.isOpened():
		sys.exit('Output file, ' + out_filename + ',  couldn\'t be created')

	while vid.isOpened():

		available, frame = vid.read()

		if available:
			if args.show_video:
				cv2.imshow('Input Video', frame)

			# Blurring helps the Otsu algorithm more easily find the threshold value to use
			frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			blurred = cv2.GaussianBlur(frame_gray, ksize=(5, 5), sigmaX=0)

			# Get the Otsu Threshold mask. 
			_, threshold = cv2.threshold(blurred, thresh=0, maxval=255, type=cv2.THRESH_OTSU) 

			# Invert so that the sea is white and sky black, makes closing easier  
			threshold = np.invert(threshold)
			
			# Closing, useful in removing small black patches in the white areas 
			closed = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel = np.ones((9,9), np.uint8))
			
			if args.show_video:
				cv2.imshow('Annotated Video', closed)

			out.write(closed)
			
		else:
			break

	vid.release()
	out.release()
	cv2.destroyAllWindows()