
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

I used python and numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12360
* The shape of a traffic sign image is (32,32,3). These are 32x32 images with 3 color channels
* The number of unique classes/labels in the data set is 43

####1. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It consists of bar charts showing how the data is distributed among the various classes. It is noted that the distribution is not even and that there are some classes with few examples.
####2. Comments
* Distribution of signs among training, test and validation seem roughly similar
* Label 38 (keep right) occurs more frequently than Label 39 (keep left)
* 30, 50, 70, 80 occurs more frequently than 20 and 60 kmph, and end of speed limit sign
![alt text][image1]

Sample images are displayed in the project notebook and the following is observed
# Comments
* Images don't appear to be clean. For the pixel sizes, they appear to be blurry
* Some dark images imply that the data set  has not been pruned.
* Signs have color patterns. However the lighting conditions vary drastically. (H,S,V) mapping usually deals with lighting variations to some extent. But in this case, the conditions vary between extereley bright and extremely dark. Grayscale appears to be a better bet even if the color information is lost.
* Converting to grayscale also seems to accentuate the pixel differences in darker images as seen below.



###Design and Test a Model Architecture

####1. Preprocessing steps consisted of conversion to grayscale and mean centering and normalization. 

Converting to grayscale helped in accentuating th epixel differences in darker images. Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because relative differences in pixel values are more important to discern shaped, not the absolute values. 

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because ... 

Here is an example of an original image and an augmented image:

![alt text][image3]

The difference between the original data set and the augmented data set is the following ... 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers, as in LeNet:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flattening         | outputs 400 |
| Fully connected		| outputs 120        									|
| RELU					|												|
| Dropout					|					keep_prob = 0.5							|
| Fully connected		| outputs 84        									|
| RELU					|												|
| Dropout					|					keep_prob = 0.5							|
| Final Fully connected		layer| outputs = 43 (number of classes)    									|
| Softmax				| Used in loss computation   									|

 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with mean cross entropy losses between the softmax probablities and the one-hot encodd labels. The following parameters were used
* Learning rate of 0.001
* Batch size of 128
* 10 EPOCHs

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.
I started with the LeNet architecture as suggested and tried optimizing various parameters. The initial model was overfitting the data since the training losses were much smaller than teh validation losses. I tried both L2 regularization techniques and dropouts and finally settled onusing dropouts after the fully connected layers.

My final model results were:
* training set accuracy of 98%
* validation set accuracy close to 94%
* test set accuracy of just over 92%

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![30kmph speed limit][https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTpWpIjjBsyfYPpUMSX19SJWO64hcTL8N9yoZlulTRQwQ0YWoRFBg] ![Slippery road][http://media.gettyimages.com/vectors/slippery-road-risk-of-ice-german-warning-sign-vector-id180585671?s=170667a] ![Bumpy road][http://storage.torontosun.com/v1/blogs-prod-photos/e/4/3/8/0/e43800d91b0c525906f0fbfb93f5b527.jpg?stmp=1290377910] 
![Children crossing][http://www.gettyimages.com/detail/photo/german-traffic-signs-royalty-free-image/465921901?esource=SEO_GIS_CDN_Redirect] ![Keep Right][http://media.gettyimages.com/photos/german-traffic-signs-picture-id459380917?s=170667a]



####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30kmph speed limit      		| 30kmph speed limit    									| 
| Slippery road     			| Slippery road										|
| Bumpy road					| Bumpy road											|
| Children crossing	      		| Children crossing					 				|
| Keep Right			| 50kmph speed limit       							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 100%. While this is less than the test set accuracy of 92%, it is statistically consistent given that only 5 images are chosen and the results are within statistical bounds of the test set accuracy. It si noted, that in some other runs not recorded here, all 5 of the mages were chosen correctly

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.


 For image  0  True label =  1 Speed limit (30km/h) 

| Probability | Prediction  |
|:---------------------:|:---------------------------------------------:|
|     99.75   | 1 Speed limit (30km/h) |
|      0.13   | 2 Speed limit (50km/h) |
|      0.12   | 0 Speed limit (20km/h) |
|      0.00   | 4 Speed limit (70km/h) |
|      0.00   | 5 Speed limit (80km/h) |

 For image  1  True label =  23 Slippery road 

| Probability | Prediction  |
|:---------------------:|:---------------------------------------------:|
|     89.88   |23        Slippery road |
|      3.50   |29    Bicycles crossing |
|      2.66   |24 Road narrows on the right |
|      2.13   |19 Dangerous curve to the left |
|      1.00   |30   Beware of ice/snow |

 For image  2  True label =  22 Bumpy road 

| Probability | Prediction  |
|:---------------------:|:---------------------------------------------:|
|     99.44   |22           Bumpy road |
|      0.26   |29    Bicycles crossing |
|      0.11   |25            Road work |
|      0.09   |18      General caution |
|      0.06   |26      Traffic signals |

 For image  3  True label =  28 Children crossing 

| Probability | Prediction  |
|:---------------------:|:---------------------------------------------:|
|     99.97   |28    Children crossing |
|      0.01   |19 Dangerous curve to the left |
|      0.01   |20 Dangerous curve to the right |
|      0.00   |23        Slippery road |
|      0.00   |11 Right-of-way at the next intersection |

 For image  4  True label =  38 Keep right 

| Probability | Prediction  |
|:---------------------:|:---------------------------------------------:|
|     90.74   | 2 Speed limit (50km/h) |
|      9.26   | 1 Speed limit (30km/h) |
|      0.00   |13                Yield |
|      0.00   | 0 Speed limit (20km/h) |
|      0.00   | 3 Speed limit (60km/h) |


