
# Traffic Sign Classifier Project

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/barchart.png "Visualization"
[image2]: ./output_images/grayscale.png "Grayscaling"
[image3]: ./output_images/gtraffic1.jpg "Traffic Sign 1"
[image4]: ./output_images/gtraffic2.jpg "Traffic Sign 2"
[image5]: ./output_images/gtraffic3.jpg "Traffic Sign 3"
[image6]: ./output_images/gtraffic4.jpg "Traffic Sign 4"
[image7]: ./output_images/gtraffic5.jpg "Traffic Sign 5"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/pnsrao/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb) and [html version](https://github.com/pnsrao/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.html) of the project code.

### Data Set Summary & Exploration

I used python and numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12360
* The shape of a traffic sign image is (32,32,3). These are 32x32 images with 3 color channels
* The number of unique classes/labels in the data set is 43

#### 1. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It consists of bar charts showing how the data is distributed among the various classes. It is noted that the distribution is not even and that there are some classes with few examples.
![alt text][image1] 
* Comments
  * Distribution of signs among training, test and validation seem roughly similar
  * 30, 50, 70, 80 occurs more frequently than 20 and 60 kmph, and end of speed limit sign

Sample images are displayed in the project notebook and the following is observed
* Comments
  * Images don't appear to be clean. For the pixel sizes, they appear to be blurry
  * Some dark images imply that the data set  has not been pruned.
  * Signs have color patterns. However the lighting conditions vary drastically. (H,S,V) mapping usually deals with lighting variations to some extent. But in this case, the conditions vary between extereley bright and extremely dark. Grayscale appears to be a better bet even if the color information is lost.
  * Converting to grayscale also seems to accentuate the pixel differences in darker images as seen later.

### Design and Test a Model Architecture

#### 1. Preprocessing steps consisted of conversion to grayscale and mean centering and normalization. 

Converting to grayscale helped in accentuating the pixel differences in darker images. Here is an example of a traffic sign image before and after converting to grayscale.

![alt text][image2]

As a last step, I normalized the image data because relative differences in pixel values are more important to discern shapes, not the absolute values. 
 
#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) 

My final model consisted of the following layers, as in LeNet:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				|
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

#### 3. Describe how you trained your model.

To train the model, I used an Adam optimizer with mean cross entropy losses between the softmax probablities and the one-hot encoded labels. The following parameters were used
* Learning rate of 0.001
* Batch size of 128
* 10 EPOCHs

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.
I started with the suggested LeNet architecture and tried optimizing various parameters. The initial model was overfitting the data since the training losses were much smaller than the validation losses. I tried both L2 regularization techniques and dropouts and finally settled on using dropouts after the fully connected layers.

My final model results were:
* training set accuracy of roughly 98%
* validation set accuracy close to 94%
* test set accuracy of roughly 92%

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web. These are cropped and resized to 32x32 images as needed by the input to the model.

![30kmph speed limit][image3] ![Slippery road][image4] ![Bumpy road][image5] 
![Children crossing][image6] ![Keep Right][image7]

Web links for the original images are as follows (as of 8/27/2017). 
[30kmph limit](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTpWpIjjBsyfYPpUMSX19SJWO64hcTL8N9yoZlulTRQwQ0YWoRFBg), [Slippery Road](http://media.gettyimages.com/vectors/slippery-road-risk-of-ice-german-warning-sign-vector-id180585671?s=170667a), [Bumpy road](http://storage.torontosun.com/v1/blogs-prod-photos/e/4/3/8/0/e43800d91b0c525906f0fbfb93f5b527.jpg?stmp=1290377910), [Children crossing](http://www.gettyimages.com/detail/photo/german-traffic-signs-royalty-free-image/465921901?esource=SEO_GIS_CDN_Redirect), [Keep Right](http://media.gettyimages.com/photos/german-traffic-signs-picture-id459380917?s=170667a).

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. 

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30kmph speed limit      		| 30kmph speed limit    									| 
| Slippery road     			| Slippery road										|
| Bumpy road					| Bumpy road											|
| Children crossing	      		| Children crossing					 				|
| Keep Right			| 30kmph speed limit       							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. While this is less than the test set accuracy of 92%, it is statistically consistent given that only 5 images are chosen and the results are within statistical bounds of the test set accuracy. It is noted, that in some other runs not recorded here, all 5 of the mages were chosen correctly

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The following data is obtained from the project notebook

 For image  0  True label =  1 Speed limit (30km/h) 

| Probability | Prediction  |
|:---------------------:|:---------------------------------------------:|
|     97.52   | 1 Speed limit (30km/h) |
|      2.16   | 0 Speed limit (20km/h) |
|      0.27   | 4 Speed limit (70km/h) |
|      0.05   | 5 Speed limit (80km/h) |
|      0.01   | 2 Speed limit (50km/h) |

 For image  1  True label =  23 Slippery road 

| Probability | Prediction  |
|:---------------------:|:---------------------------------------------:|
|     99.19   |23        Slippery road |
|      0.68   |20 Dangerous curve to the right |
|      0.08   |19 Dangerous curve to the left |
|      0.06   |30   Beware of ice/snow |
|      0.00   |24 Road narrows on the right |

 For image  2  True label =  22 Bumpy road 

| Probability | Prediction  |
|:---------------------:|:---------------------------------------------:|
|     84.71   |22           Bumpy road |
|      6.95   |29    Bicycles crossing |
|      5.56   |28    Children crossing |
|      1.74   |36 Go straight or right |
|      0.40   |25            Road work |

 For image  3  True label =  28 Children crossing 

| Probability | Prediction  |
|:---------------------:|:---------------------------------------------:|
|     85.88   |28    Children crossing |
|      6.68   |24 Road narrows on the right |
|      4.42   |29    Bicycles crossing |
|      1.03   |20 Dangerous curve to the right |
|      0.52   |23        Slippery road |

 For image  4  True label =  38 Keep right 

| Probability | Prediction  |
|:---------------------:|:---------------------------------------------:|
|     90.12   | 1 Speed limit (30km/h) |
|      9.74   | 2 Speed limit (50km/h) |
|      0.08   |19 Dangerous curve to the left |
|      0.02   |31 Wild animals crossing |
|      0.02   |39            Keep left |

The first two predictions appear to be strong ones. The last three signs had relatively lower confidence. However only the "Keep Right" sign was classified incorrectly.
