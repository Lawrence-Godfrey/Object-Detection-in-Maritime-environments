import cv2
import numpy as np
import os
folder = 'Data/MODD2/video_data/kope67-00-00004500-00005050/frames/'
filenames = sorted(os.listdir(folder))

print(sorted([os.path.join(folder,filename) for filename in filenames]))

frames = []
for filename in filenames:
	frame = cv2.imread(os.path.join(folder,filename))
	frames.append(frame)


frameRate = 25

for i in range(0, len(frames)-1, 2):
	cv2.imshow('Video Left', frames[i])
	cv2.imshow('Video Right', frames[i+1])


	if cv2.waitKey(1000//frameRate) == 27:
		break

cv2.destroyAllWindows()