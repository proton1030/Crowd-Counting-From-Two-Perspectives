# Crowd Counting From Two Viewpoints Using Neural Network

- Images (3000+) captured from two different angles and labeled with crowd quantity using a MATLAB GUI application.
- Using OpenCV to do preprocessing like background subtraction, Canny edge detection and etc. on images.
- Designed several novel holistic features and train a feed forward neural network in Tensorflow to generate crowd size predictions.



![etc](https://raw.githubusercontent.com/proton1030/Crowd-Counting-From-Two-Perspectives/master/etc/poster.jpg)



## Files Description

The program requires Tensorflow GPU version to run.

The main executable file is final.py. Type "python final.py" to run the code and see the results.

The csv files consists of features extracted from the database. The database was not included since it's size's about 1.5 G. Therefore the data_gathering_L() and data_gathering_R() cannot be run. However, it can still use the csv files to train the network.

I've also recorded a video "demo.mov" of my desktop running the code just in case

The "data_labeling" folder consists of the MATLAB GUI used for labeling the ground truth. I only put 10 images in there so that it could work. You initially assign the ground truth on the first frame and frames after the current frame will sync with this frame's ground truth.
