import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import scipy.io as spio
import csv
from matplotlib import pyplot as plt
import scipy.signal as sig
import cv2
import os

def read_ground_truth():
    return spio.loadmat('./gt.mat', squeeze_me=True)

def data_read(LorR):
    # Reads data which consists of features
    if LorR:
        x = np.genfromtxt('input_pixel_L.csv', delimiter=' ')
    else:
        x = np.genfromtxt('input_pixel_R.csv', delimiter=' ')
    y = np.genfromtxt('output_count.csv', delimiter=' ')
    y = np.reshape(y, (-1, 1))
    return x,y

def data_gathering_L(): # This is not runable since I didn't include the image files
    fgbg = cv2.createBackgroundSubtractorKNN(history=600, detectShadows=True) # Background subtractor
    crowdData = np.array(read_ground_truth()['crowd'])
    crowdData = np.reshape(crowdData, (-1, 1))
    pixelSumData = np.array([])
    flag = True

    relDataPath = '../dataset/img_series/3_L_img/'

    # Points for subtle perspective transformation
    pts1_lf = np.float32([[137,536],[296,531],[98,796],[540,770]])
    pts2_lf = np.float32([[137,536],[296,536],[100,796],[540,796]])
    M1 = cv2.getPerspectiveTransform(pts1_lf,pts2_lf)

    for i in range(0, 3286):

        frame = cv2.imread(relDataPath+'3_L_' + format(i, '04') + '.jpg')
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        roi_pts = np.int32([[137,536],[296,531],[540,727],[540,770],[98,796]]) # Mask points
        roi_mask = np.zeros_like(frame_gray)
        cv2.fillConvexPoly(roi_mask, roi_pts, (1,1,1))
        
        frame_median = cv2.medianBlur(frame_gray,5)
        frame_canny = cv2.Canny(frame_median, 50, 150) #Get edges
        frame_bg = fgbg.apply(frame_median) # Get foreground
        frame_bg = cv2.medianBlur(frame_bg,5)
        frame_mask = np.uint8(np.where(frame_bg > 130.0, 1, 0) * frame_canny) #Get edges*foreground
        frame_bg = np.uint8(np.where(frame_bg > 0, 1, 0))
        frame_mask_roi = frame_mask * roi_mask

        frame_mask_roi = cv2.warpPerspective(frame_mask_roi, M1, (540,960))
        frame_bg_roi = cv2.warpPerspective(frame_bg, M1, (540,960))
        cropped_roi = frame_mask_roi[536:796,100:540] #Crop image size to the size just fit
        cropped_bg_roi = frame_bg_roi[536:796,100:540]
        cropped_roi = np.uint8(np.where(cropped_roi > 0, 1, 0))
        cropped_bg_roi = np.uint8(np.where(cropped_bg_roi > 0, 1, 0))

        _, edge_contours, _ = cv2.findContours(cropped_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) #Getting Contour quantities
        _, blob_contours, _ = cv2.findContours(cropped_bg_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        perspective_corr = 105.0 / np.linspace(31.0, 105.0, num=260)
        row_sum = np.sum(cropped_roi, axis=1)
        col_sum = np.sum(cropped_roi, axis=0)
        row_bg_sum = np.sum(cropped_bg_roi, axis=1)
        col_bg_sum = np.sum(cropped_bg_roi, axis=0)
        row_sum = row_sum.astype("float64") * perspective_corr
        row_bg_sum = row_bg_sum.astype("float64") * perspective_corr

        x = 13  # 20 bins * 13 rows per bin = 260 rows
        y = 22  # 20 bins * 22 cols per bin = 440 cols
        bins = 20

        edge_pixel_rows = np.array([np.sum(row_sum[x*n:x*n+x-1]) for n in range(0 , bins)])
        edge_pixel = np.array(np.concatenate(([np.sum(row_sum[x*n:x*n+x-1]) for n in range(0 , bins)] , \
                                        [np.sum(col_sum[y*n:y*n+y-1]) for n in range(0 , bins)])))
        agg_bg_edge_pixel = np.array(np.concatenate(([np.sum(row_sum[x*n:x*n+x-1]) for n in range(0 , bins)] , \
                                        [np.sum(col_sum[y*n:y*n+y-1]) for n in range(0 , bins)], \
                                        [np.sum(row_bg_sum[x*n:x*n+x-1]) for n in range(0 , bins)] , \
                                        [np.sum(col_bg_sum[y*n:y*n+y-1]) for n in range(0 , bins)])))
        edgePixelRows_edgeContour_blobContour = np.append(edge_pixel_rows, [len(edge_contours), len(blob_contours)])
        data = edgePixelRows_edgeContour_blobContour
        if flag:
            pixelSumData = np.array([data])
            flag = False
        else:
            pixelSumData = np.vstack((pixelSumData, data))
    

    print crowdData.shape
    print pixelSumData.shape

    example=csv.writer(open('input_pixel_L.csv', 'wb'), delimiter=' ')
    example.writerows(pixelSumData[4:])

    example=csv.writer(open('output_count.csv', 'wb'), delimiter=' ')
    example.writerows(crowdData[4:])

def data_gathering_R(): # This is not runable since I didn't include the image files, everything's same to data_gathering_L()
    fgbg = cv2.createBackgroundSubtractorKNN(history=600, detectShadows=True)
    crowdData = np.array(read_ground_truth()['crowd'])
    crowdData = np.reshape(crowdData, (-1, 1))
    pixelSumData = np.array([])
    flag = True

    relDataPath = '../dataset/img_series/3_R_img/'

    pts1_lf = np.float32([[222,632],[388,635],[0,892],[269,912]])
    pts2_lf = np.float32([[222,635],[388,635],[0,895],[269,895]])
    M1 = cv2.getPerspectiveTransform(pts1_lf,pts2_lf)

    for i in range(0, 3286):

        frame = cv2.imread(relDataPath+'3_R_' + format(i, '04') + '.jpg')
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        roi_pts = np.int32([[222,632],[388,635],[269,912],[0,892],[0,760]])
        roi_mask = np.zeros_like(frame_gray)
        cv2.fillConvexPoly(roi_mask, roi_pts, (1,1,1))
        
        frame_median = cv2.medianBlur(frame_gray,5)
        frame_canny = cv2.Canny(frame_median, 50, 150)
        frame_bg = fgbg.apply(frame_median)
        frame_bg = cv2.medianBlur(frame_bg,5)
        frame_mask = np.uint8(np.where(frame_bg > 130.0, 1, 0) * frame_canny)
        frame_bg = np.uint8(np.where(frame_bg > 0, 1, 0))
        frame_mask_roi = frame_mask * roi_mask

        frame_mask_roi = cv2.warpPerspective(frame_mask_roi, M1, (540,960))
        frame_bg_roi = cv2.warpPerspective(frame_bg, M1, (540,960))
        cropped_roi = frame_mask_roi[635:895,0:380]
        cropped_bg_roi = frame_bg_roi[635:895,0:380]
        cropped_roi = np.uint8(np.where(cropped_roi > 0, 1, 0))
        cropped_bg_roi = np.uint8(np.where(cropped_bg_roi > 0, 1, 0))

        _, edge_contours, _ = cv2.findContours(cropped_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        _, blob_contours, _ = cv2.findContours(cropped_bg_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  

        perspective_corr = 84.0 / np.linspace(32.0, 84.0, num=260)
        row_sum = np.sum(cropped_roi, axis=1)
        col_sum = np.sum(cropped_roi, axis=0)
        row_bg_sum = np.sum(cropped_bg_roi, axis=1)
        col_bg_sum = np.sum(cropped_bg_roi, axis=0)
        row_sum = row_sum.astype("float64") * perspective_corr
        row_bg_sum = row_bg_sum.astype("float64") * perspective_corr

        x = 13  # 20 * 13 bins = 260 rows
        y = 19  # 20 * 19 bins = 380 cols
        bins = 20
        edge_pixel_rows = np.array([np.sum(row_sum[x*n:x*n+x-1]) for n in range(0 , bins)])
        bg_pixel_rows = np.array([np.sum(row_bg_sum[x*n:x*n+x-1]) for n in range(0 , bins)])
        edge_pixel = np.array(np.concatenate(([np.sum(row_sum[x*n:x*n+x-1]) for n in range(0 , bins)] , \
                                        [np.sum(col_sum[y*n:y*n+y-1]) for n in range(0 , bins)])))
        agg_bg_edge_pixel = np.array(np.concatenate(([np.sum(row_sum[x*n:x*n+x-1]) for n in range(0 , bins)] , \
                                        [np.sum(col_sum[y*n:y*n+y-1]) for n in range(0 , bins)], \
                                        [np.sum(row_bg_sum[x*n:x*n+x-1]) for n in range(0 , bins)] , \
                                        [np.sum(col_bg_sum[y*n:y*n+y-1]) for n in range(0 , bins)])))
        edgePixelRows_edgeContour_blobContour = np.append(edge_pixel_rows, [len(edge_contours), len(blob_contours)])
        data = edgePixelRows_edgeContour_blobContour

        if flag:
            pixelSumData = np.array([data])
            flag = False
        else:
            pixelSumData = np.vstack((pixelSumData, data))
    

    print crowdData.shape
    print pixelSumData.shape

    example=csv.writer(open('input_pixel_R.csv', 'wb'), delimiter=' ')
    example.writerows(pixelSumData[4:])

    example=csv.writer(open('output_count.csv', 'wb'), delimiter=' ')
    example.writerows(crowdData[4:])




#------------------------------------------Training stuff------------------------------------------------------------------------------------

def preprocess_data(input , output):
    #split dataset into two proportions
    return train_test_split(input, output, test_size=0.95)

def weight_variable(shape):
     initial = tf.truncated_normal(shape, stddev=0.1)
     return tf.Variable(initial)

def bias_variable(shape):
     initial = tf.constant(0.1, shape=shape)
     return tf.Variable(initial)

def train(inputData, outputData):

    train_X, test_X, train_y, test_y = preprocess_data(inputData, outputData)

    # Layer's sizes
    x_size = train_X.shape[1]
    y_size = train_y.shape[1]

    #Network params
    episodes = 30
    batch_size = 1
    hidden_units = x_size
    learning_rate = 0.00008

    # Symbols
    x = tf.placeholder("float", shape=[None, x_size])
    y_ = tf.placeholder("float", shape=[None, y_size])

    # Input Layer
    W1 = weight_variable([x_size, hidden_units])
    b1 = bias_variable([hidden_units])
    r1 = tf.nn.relu(tf.matmul(x, W1) + b1)

    # 4 Hidden Layers
    W2 = weight_variable([hidden_units, hidden_units])
    b2 = bias_variable([hidden_units])
    r2 = tf.nn.relu(tf.matmul(r1, W2) + b2)

    W3 = weight_variable([hidden_units, hidden_units])
    b3 = bias_variable([hidden_units])
    r3 = tf.nn.relu(tf.matmul(r2, W3) + b3)

    W4 = weight_variable([hidden_units, hidden_units])
    b4 = bias_variable([hidden_units])
    r4 = tf.nn.relu(tf.matmul(r3, W4) + b4)

    # Output Layer
    W5 = weight_variable([hidden_units, y_size])
    b5 = bias_variable([1])
    y = tf.nn.relu(tf.matmul(r4,W5)+b5) 

    # Uses MSE as cost function
    mean_square_error = tf.reduce_sum(tf.square(y-y_))
    training = tf.train.AdamOptimizer(learning_rate).minimize(mean_square_error)

    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for epoch in range(episodes):
        error = np.array([])
        for i in range(train_X.shape[0]-batch_size+1):
            _, current_error = sess.run([training, mean_square_error],  feed_dict={x: train_X[i:i+batch_size], y_:train_y[i:i+batch_size]})
            error = np.append(error, current_error)
        print("Epoch = %d, mean error = %.2f, MSE error = %.2f" % (epoch + 1, np.mean(error), (error**2).mean(axis=0)))

    # Test phase
    estimated_quantity = np.array([])
    ground_truth_quantity = np.array([])
    for i in range(inputData.shape[0]-batch_size+1):
        current_output = sess.run([y],  feed_dict={x: inputData[i:i+batch_size]})
        estimated_quantity = np.append(estimated_quantity , np.round(current_output[0][0][0]))
        ground_truth_quantity = np.append(ground_truth_quantity , outputData[i])
    writer=csv.writer(open('test_result.csv', 'wb'), delimiter=' ')
    writer.writerows([estimated_quantity,ground_truth_quantity])

    sess.close()


def evaluation(plotFlag):
    data = np.genfromtxt('test_result.csv', delimiter=' ')
    data[0] = sig.medfilt(data[0],41) # Median filtering on the output
    if plotFlag:
        gt_plt = plt.plot(data[1])
        est_plt = plt.plot(data[0])
        plt.show()
    error = data[0] - data[1]
    print ("MSE test data = %.4f, acceptability = %.2f" % ((error**2).mean(axis=0) , np.sum(np.abs(error) <= 3)/float(len(error))*100.0))

    

#------------------------------------------Main------------------------------------------------------------------------------------  

def main() :
    # These are feature extraction functions, however they can't be run due to 
    # the lack of the image database. I've stored one set of features for each 
    # viewpoints in the correpsonding .csv files

    # data_gathering_L() 
    # data_gathering_R()

    # Read features from csv files
    pixelSumData_L , crowdData = data_read(True)
    pixelSumData_R , crowdData = data_read(False)
    contour_diff = pixelSumData_L[:,20:] - pixelSumData_R[:,20:]
    pixelSumData_L = np.hstack((pixelSumData_L , contour_diff))
    pixelSumData_R = np.hstack((pixelSumData_R , contour_diff))

    for i in range(0,1):
        train(pixelSumData_L , crowdData) #Can also change input to pixelSumData_R
        evaluation(1) # 1 for plotting estimation curves


if __name__ == '__main__':
    main()