### Example Motion Detection using difference between current and previous frame

code in [`BasicMotionDetection.py`](BasicMotionDetection.py)

![alt-text](https://imgur.com/SxPqI3K.gif)
![alt-text](https://imgur.com/qkizkBx.gif)

---

### Example using K Nearest Neighbours Background/Foreground Segmentation Algorithm
Described in : "Zoran Zivkovic and Ferdinand van der Heijden. Efficient adaptive density estimation per image pixel for the task of background subtraction. Pattern recognition letters, 27(7):773–780, 2006."

code in [`BackgroundSubtraction.py`](BackgroundSubtraction.py)

![alt-text](https://imgur.com/SxPqI3K.gif)
![alt-text](https://imgur.com/rEtOy5C.gif)

---

### Example using Mean Averaging Background Subtraction 
Here the last 50 frames are averaged. This average is used as an approximation of the background
Clearly works better for moving background, however, the objects also need to be moving fast, otherwise they will be approximated as background

Code in [`MeanBackroundSubtraction.py`](MeanBackroundSubtraction.py)

![alt-text](https://imgur.com/QP8jQFc.gif)
![alt-text](https://imgur.com/QQL35Yn.gif)


