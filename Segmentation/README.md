## Segmentation using Deep Learning Segmentation Models
These are examples of segmentation using pre-trained segmentation models to classify regions of the maritime video datasets. 


### Sea/Sky Segmentation using the UNet segmentation model
This model uses the VGG19 backbone, with the fully-connected layers removed, in a encoder-decoder configuration. 
The model is trained on the single video from the Singapore Martime Dataset shown below, using Otsu Thresholding to automatically annotate the video.

![alt-text](https://imgur.com/wYsIsUL.gif)

Which results in a validation IoU of 0.9786

Applying the model to a different video from the same dataset results in the output shown below. 

![alt-text](https://imgur.com/FgH4dX6.gif)
![alt-text](https://imgur.com/DUwv7Le.gif)