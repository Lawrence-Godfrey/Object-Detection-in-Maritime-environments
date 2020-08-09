### Example Motion Detection using difference between current and previous frame

code in [`FrameDifferencing.py`](FrameDifferencing.py)

![alt-text](https://imgur.com/GNsaPOe.gif)
![alt-text](https://imgur.com/CmtLS5T.gif)

---

### Example using K Nearest Neighbours Background/Foreground Segmentation Algorithm
Described in : "Zoran Zivkovic and Ferdinand van der Heijden. Efficient adaptive density estimation per image pixel for the task of background subtraction. Pattern recognition letters, 27(7):773–780, 2006."

code in [`BackgroundSubtraction.py`](BackgroundSubtraction.py)

![alt-text](https://imgur.com/GNsaPOe.gif)
![alt-text](https://imgur.com/H7WfHFT.gif)

---

### Example using Mean Averaging Background Subtraction 
Here the last 50 frames are averaged. This average is used as an approximation of the background
Clearly works better for moving background, however, it is more computationally expensive and memory-hungry.

Code in [`MeanBackgroundSubtraction.py`](MeanBackroundSubtraction.py)

![alt-text](https://imgur.com/GNsaPOe.gif)
![alt-text](https://imgur.com/QWscBo3.gif)


---

### Example using Median Averaging Background Subtraction 
Same as previous except using median of last N frames. 

Code in [`MedianBackgroundSubtraction.py`](MedianBackgroundSubtraction.py)

![alt-text](https://imgur.com/GNsaPOe.gif)
![alt-text](https://imgur.com/JaeVDTy.gif)


--- 

### Example using Median Approximation Background Subtraction

Here the background is initally black, then, for every frame, the pixels in the background which have a lower value than the corresponding pixel in the current frame are increased by 1, and the pixels which have a higher value than the current frame are decreased by 1. 

The background converges to an approximation of the actual background

Code in [`MedianApproximationBackgroundSubtraction.py`](MedianApproximationBackgroundSubtraction.py)

![alt-text](https://imgur.com/GNsaPOe.gif)
![alt-text](https://imgur.com/JaeVDTy.gif)

--- 

### Example using a Running Gaussian Average
Here a Gaussian Probabilty Density Function (PDF) is fit onto each pixel. 
In order to avoid fitting the pdf from scratch at each new frame time, t, a running (or on-line cumulative) average is computed using a learning rate.

A pixel is classified as being part of the background if it is within T standard deviations of the mean. 

This was first proposed in “Pfinder: real-time tracking of the human body” - C. Wren, A. Azarhayejani, T. Darrell, and A.P. Pentland

You can see that it struggles to model both the water and the buildings in the same PDF, hence the need for the Mixture Of Gaussians approach. 

Code in [`RunningGaussianAverage.py`](RunningGaussianAverage.py)

![alt-text](https://imgur.com/GNsaPOe.gif)
![alt-text](https://imgur.com/aqtr7GD.gif)

--- 

### Example using Mixture Of Gaussians (MOG)
This uses the builtin openCV MOG function 

Described in "Adaptive background mixture models for real-time tracking" - Chris Stauffer and W.E.L Grimson

Here there are multiple Gaussian distributions, each adapted over time. 
This allows for multiple types of backgrounds to be modelled 

Code in [`BackgroundSubtraction.py`](BackgroundSubtraction.py)

![alt-text](https://imgur.com/GNsaPOe.gif)
![alt-text](https://imgur.com/bzfYIf1.gif)

--- 