# Exercise 3 (Training and Creating a model) report

## The Network
A simplified version of the PoseNet is used. The training of the PoseNet need around 15 Minutes per epoch. By omitting one of the its three "inception" modules, the calculation time has been decreased to around 11 Minutes and the used memory is less, while the performance stays at a similar level. 

## The loss function 
The loss function is the sum of the position loss and the quaternion loss that is additionally weighted by the factor Beta.

## Training
There are severeal runs with various values for beta:
For beta = 700, beta = 1000 and beta = 2000 the runs are stopped early because of increasing errors after 10-15 epochs. 

A run with beta = 250 is finished over 200 Epochs and reached a median error of 4.07 m, 2.87 °. 

---> Picture 

Another run with beta = 400 reached a median error of 4.28 m, 3.03 ° after 50 epochs. The training is going on. 


## Implementation
First, the model is trained without pretrained weightes because they are not found online for pytorch. 
After several abortions of the training procedure caused by reboots of the pool computers, the model is saved after each epoch. So the training can be continued at every point. 