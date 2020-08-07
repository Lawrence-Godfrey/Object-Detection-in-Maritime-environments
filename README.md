### Example Motion Detection using difference between current and previous frame

code in [`FrameDifferencing.py`](FrameDifferencing.py)

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

Code in [`MeanBackgroundSubtraction.py`](MeanBackroundSubtraction.py)

![alt-text](https://imgur.com/hujnY43.gif)
![alt-text](https://imgur.com/sgn7KTt.gif)


---

### Example using Median Averaging Background Subtraction 

Code in [`MedianBackgroundSubtraction.py`](MedianBackgroundSubtraction.py)

![alt-text](https://imgur.com/myy74zS.gif)
![alt-text](https://imgur.com/wyTRH1c.gif)
