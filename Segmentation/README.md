## Segmentation using Deep Learning Segmentation Models
These are examples of segmentation using pre-trained segmentation models to classify regions of the maritime video datasets. 


### Sea/Sky Segmentation using the UNet segmentation model
This model uses the VGG19 backbone, with the fully-connected layers removed, in a encoder-decoder configuration. 
The model is trained on the single video from the Singapore Martime Dataset shown below, using Otsu Thresholding to automatically create a mask for the video.

![alt-text](https://imgur.com/wYsIsUL.gif)

Which results in a validation **IoU** of **0.9786**

Applying the model to a different video from the same dataset results in the output shown below. 

![alt-text](https://imgur.com/uEh7f1l.gif)
![alt-text](https://imgur.com/saZPeYE.gif)

---

### Making Custom Masks for Training
In order to train the segmentation model with multiple classes I needed to create custom maskes which track objects in the video
To start, I masked the video with 4 colours for 4 classes: sea, sky, boats, and other. 

Below is an example of the input video and corresponding mask:

![alt-text](https://imgur.com/Xwwb5k5.gif)
![alt-text](https://imgur.com/ha89aAs.gif)

---

### Multi-Class Segmentation 
Below is the prediction of the segmentation model trained on three vides from the Singapore Maritime Dataset. 

![alt-text](https://imgur.com/zdTul7T.gif)
![alt-text](https://imgur.com/6MKg1Lz.gif)
