### Horizon Detection using Otsu Thresholding
This method uses Otsu Thresholding to seperate the sea and sky into two segments, and inferring the horizon from there. 
Otsu's method determines an optimal global threshold value from the image histogram. This works well in images where there are two distinct groups of value, which is why it works well in sea/sky segmentation. 

Since the line is based off of only the pixels at the edges of the frame, it will be highly inaccurate when object like ships are present at the edges of the frame.

Code can be found in [`OtsuHorizonDetection.py`](OtsuHorizonDetection.py)

![alt-text](https://imgur.com/wYsIsUL.gif)
![alt-text](https://imgur.com/09Gu1bD.gif)

---

### Horizon Detection using variances
This is a simplification of a method introduced in "Vision-guided flight stability and control for micro air vehicles" - S. M. Ettinger, M. C. Nechyba, P. G. Ifju, and M. Waszak

The predicted horizon line is moved across the frame in small increments. For every increment, a cost function is evaluated based on the variance of the pixels below the line and the variance of the pixels above the line. This cost function will obviously be lowest when the line perfectly divides the sea and sky.  

While this does increase the accuracy, it requires a huge amount of computation per frame. This computation can be reduced by increasing the amount by which the line is moved each step, but this does decrease the accuracy. 

Code can be found in [`VarianceHorizonDetection.py`](VarianceHorizonDetection.py)

![alt-text](https://imgur.com/wYsIsUL.gif)
![alt-text](https://imgur.com/vGCqXZf.gif)

---
