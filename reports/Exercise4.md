# Exercise 4 report
During the last weeks a visualization of the true and predicted pose is implemented. 
Furthermore, a second dataset, which serves a larger area, was preprocessed for the subsequent training.
Also, a third dataset was recorded with the test-car.
In the end, the training was executed at the google cloud platform. 

## The visualization
At first, a single pose with the UTM-Coordinates and the steering angle was visualized on google maps. 
A mode that draws the driven way of the car was added afterwards. A direct and visual comparison between the predicted and true pose was implemented as last step. 

## The new / second dataset
The given dataset was structured totally different to the first one and owns six instead of one image per pose. Thus, a lot of programmed modules had to be adapted and the dataset needed to be restructured and downsized by converting the images to jpeg.

## Training a model
Since the new dataset owns six images instead of one, it was not possible to use the pretrained model of the previous given dataset. Therefore, various values for the beta in the loss function were tried out during training. The best model reached a median validation error of 2.45 m and 1.29* and was trained with beta = 100. When it has converged, the training was continued with beta = 1 to improve the ground truth's error. 

![](images/model_100.0_2018_6_26_9_22.loss.png)

## Perfomance of the Google Cloud Platform
The given VM performs a bit slower (~1.5 min / epoch) than the pool computers(~1.2 min / epoch). 
Since there are many problems through automatic server restarts and shutdowns of the pool computers, the training on the cloud is prefered. A VM with SSD should improve the calculation time appreciable and is recommended.  

