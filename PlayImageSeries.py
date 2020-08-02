import cv2
import numpy as np
import os
folder = 'Data/MASTR1325/Images/new'
filenames = sorted(os.listdir(folder))

# print(sorted([os.path.join(folder,filename) for filename in filenames]))

frames = []
for filename in filenames:
	frame = cv2.imread(os.path.join(folder,filename))
	frames.append(frame)


frameRate = 2

for frame in frames:
	cv2.imshow('Video', frame)


	if cv2.waitKey(1000//frameRate) == 27:
		break

cv2.destroyAllWindows()