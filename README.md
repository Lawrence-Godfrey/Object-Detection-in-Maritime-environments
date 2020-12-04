 ## Modelling Water A Comparative Evaluation of Approaches to Modelling Water in Video
 This is the code repository for [this](https://github.com/Lawrence-Godfrey/Object-Detection-in-Maritime-environments/blob/master/Modelling%20Water%20A%20Comparative%20Evaluation%20of%20Approaches%20to%20Modelling%20Water%20in%20Video.pdf) final year paper. 
 
### Abstract 
Traditional object detection methods do not perform well in maritime environments, due to the complex and continuously changing background that water creates. This paper provides
a comparative evaluation between two broad categories of methods for object detection, namely learning and non-learning, to show that the learning based methods can much
more accurately detect objects in these environments. The non-learning based methods are tested using a number of background subtraction techniques, while the learning based
methods are tested using fully convolutional networks.


### [Background Subtraction Methods](Background_Subtraction/)
Background subtraction is an area of computer vision dedicated to detect moving objects
in video. As the name suggests, this requires a model for the background to be estimated,
and then subtracted from images in the video sequence to produce a mask of the moving
objects.

![Imgur](https://imgur.com/bzfYIf1.gif)

---

### [Horizon Detection Methods](Horizon_Detection/)
![Imgur](https://imgur.com/A08i7Jn.gif)

---

### [Semantic Segmentation](Segmentation/)
Semantic segmentation involves the labelling of each pixel in an image with a corresponding
class, as opposed to classifying the entire image. Therefore, the output of a semantic
segmentation model has the same width and height as the input image, with the number
of channels corresponding to the number of classes the image is being segmented into.

![Imgur](https://imgur.com/ydPcZpf)
 
--- 

### [Semantic Segmentation using Temporal Features](Temporal_Segmentation/)
In the past CNNs with 3D convolutions have been used for action recognition tasks, where a video sequence is classified as one of a number of actions, and
recently this has been extended to segmentation tasks. The approach I have taken
differs to these slightly in that it focuses on small changes over time, with just enough
temporal information to capture the motion of water, as opposed to action recognition which
generally requires a larger number of frames to capture the full range of an action.
