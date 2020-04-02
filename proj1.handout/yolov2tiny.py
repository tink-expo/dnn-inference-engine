import copy
import os
import sys
import pickle
import numpy as np
import tensorflow as tf

class YOLO_V2_TINY(object):

    # Constants.
    k_bn_epsilon = 1e-5
    k_alpha = 0.1

    def __init__(self, in_shape, weight_pickle, proc="cpu"):
        self.g = tf.Graph()
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.compat.v1.Session(config=config, graph=self.g)
        self.proc = proc
        self.weight_pickle = weight_pickle
        self.input_tensor, self.tensor_list = self.build_graph(in_shape)

    def __del__(self):
        self.sess.close()

    def build_graph(self, in_shape):
        #
        # This function builds a tensor graph. Once created,
        # it will be used to inference every frame.
        #

        # Load weight parameters from a pickle file.
        with open(self.weight_pickle, 'rb') as weight_f:
            weight_list = pickle.load(weight_f)

        # Create an empty list for tensors.
        tensor_list = []

        # Use self.g as a default graph. Refer to this API.
        ## https://www.tensorflow.org/api_docs/python/tf/Graph#as_default
        # Then you need to declare which device to use for tensor computation. The device info
        # is given from the command line argument and stored somewhere in this object.
        # In this project, you may choose CPU or GPU. Consider using the following API.
        ## https://www.tensorflow.org/api_docs/python/tf/Graph#device
        # Then you are ready to add tensors to the graph. According to the Yolo v2 tiny model,
        # build a graph and append the tensors to the returning list for computing intermediate
        # values. One tip is to start adding a placeholder tensor for the first tensor.
        # (Use 1e-5 for the epsilon value of batch normalization layers.)
        with self.g.as_default(), tf.device("/cpu:0" if self.proc == "cpu" else "/gpu:0"):
        
            x = tf.compat.v1.placeholder(tf.float32, shape=in_shape)
            input_tensor = x

            def conv_bn_layer(x, weight_dict, p=False):
                kernel = weight_dict['kernel']
                biases = weight_dict['biases']
                variance = weight_dict['moving_variance']
                gamma = weight_dict['gamma']
                mean = weight_dict['moving_mean']
                if p:
                    print(biases)
                    print()
                for i in range(len(biases)):
                    norm_factor = gamma[i] / np.sqrt(variance[i] + self.k_bn_epsilon)
                    kernel[:,:,:,i] = kernel[:,:,:,i] * norm_factor
                    biases[i] = biases[i] * norm_factor - mean[i] * norm_factor
                if p:
                    print(biases)

                x = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
                x = tf.add(x, biases)
                x = tf.nn.leaky_relu(x, self.k_alpha)

                tensor_list.append(x)
                return x

            def maxpool_layer(x, kernel_size, stride_size, padding='VALID'):
                x = tf.nn.max_pool2d(
                        x, 
                        ksize=[1, kernel_size, kernel_size, 1], 
                        strides=[1, stride_size, stride_size, 1], 
                        padding=padding)

                tensor_list.append(x)
                return x

            def conv_layer(x, weight_dict):
                kernel = weight_dict['kernel']
                biases = weight_dict['biases']
                x = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='SAME')
                x = tf.add(x, biases)

                tensor_list.append(x)
                return x

            x = conv_bn_layer(x, weight_list[0])
            x = maxpool_layer(x, 2, 2)

            x = conv_bn_layer(x, weight_list[1])
            x = maxpool_layer(x, 2, 2)

            x = conv_bn_layer(x, weight_list[2])
            x = maxpool_layer(x, 2, 2)

            x = conv_bn_layer(x, weight_list[3])
            x = maxpool_layer(x, 2, 2)
            
            x = conv_bn_layer(x, weight_list[4])
            x = maxpool_layer(x, 2, 2)
            
            x = conv_bn_layer(x, weight_list[5])
            x = maxpool_layer(x, 2, 1, padding='SAME')
            
            x = conv_bn_layer(x, weight_list[6])
            
            x = conv_bn_layer(x, weight_list[7])
            
            x = conv_layer(x, weight_list[8])

        # Return the start tensor and the list of all tensors.
        return input_tensor, tensor_list

    def inference(self, img):
        feed_dict = { self.input_tensor: img }
        out_tensors = self.sess.run(self.tensor_list, feed_dict)
        return out_tensors

    
#
# Codes belows are for postprocessing step. Do not modify. The postprocessing 
# function takes an input of a resulting tensor as an array to parse it to
# generate the label box positions. It returns a list of the positions which
# composed of a label, two coordinates of left-top and right-bottom of the box
# and its color.
#

def postprocessing(predictions):

    n_classes = 20
    n_grid_cells = 13
    n_b_boxes = 5
    n_b_box_coord = 4
  
    # Names and colors for each class
    classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
    colors = [(254.0, 254.0, 254), (239.88888888888889, 211.66666666666669, 127), 
              (225.77777777777777, 169.33333333333334, 0), (211.66666666666669, 127.0, 254),
              (197.55555555555557, 84.66666666666667, 127), (183.44444444444443, 42.33333333333332, 0),
              (169.33333333333334, 0.0, 254), (155.22222222222223, -42.33333333333335, 127),
              (141.11111111111111, -84.66666666666664, 0), (127.0, 254.0, 254), 
              (112.88888888888889, 211.66666666666669, 127), (98.77777777777777, 169.33333333333334, 0),
              (84.66666666666667, 127.0, 254), (70.55555555555556, 84.66666666666667, 127),
              (56.44444444444444, 42.33333333333332, 0), (42.33333333333332, 0.0, 254), 
              (28.222222222222236, -42.33333333333335, 127), (14.111111111111118, -84.66666666666664, 0),
              (0.0, 254.0, 254), (-14.111111111111118, 211.66666666666669, 127)]
  
    # Pre-computed YOLOv2 shapes of the k=5 B-Boxes
    anchors = [1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52]
  
    thresholded_predictions = []
  
    # IMPORTANT: reshape to have shape = [ 13 x 13 x (5 B-Boxes) x (4 Coords + 1 Obj score + 20 Class scores ) ]
    predictions = np.reshape(predictions, (13, 13, 5, 25))
  
    # IMPORTANT: Compute the coordinates and score of the B-Boxes by considering the parametrization of YOLOv2
    for row in range(n_grid_cells):
      for col in range(n_grid_cells):
        for b in range(n_b_boxes):
  
          tx, ty, tw, th, tc = predictions[row, col, b, :5]
  
          # IMPORTANT: (416 img size) / (13 grid cells) = 32!
          # YOLOv2 predicts parametrized coordinates that must be converted to full size
          # final_coordinates = parametrized_coordinates * 32.0 ( You can see other EQUIVALENT ways to do this...)
          center_x = (float(col) + sigmoid(tx)) * 32.0
          center_y = (float(row) + sigmoid(ty)) * 32.0
  
          roi_w = np.exp(tw) * anchors[2*b + 0] * 32.0
          roi_h = np.exp(th) * anchors[2*b + 1] * 32.0
  
          final_confidence = sigmoid(tc)
  
          # Find best class
          class_predictions = predictions[row, col, b, 5:]
          class_predictions = softmax(class_predictions)
  
          class_predictions = tuple(class_predictions)
          best_class = class_predictions.index(max(class_predictions))
          best_class_score = class_predictions[best_class]
   
          # Flip the coordinates on both axes
          left   = int(center_x - (roi_w/2.))
          right  = int(center_x + (roi_w/2.))
          top    = int(center_y - (roi_h/2.))
          bottom = int(center_y + (roi_h/2.))
          
          if( (final_confidence * best_class_score) > 0.3):
            thresholded_predictions.append([[left, top, right, bottom], final_confidence * best_class_score, classes[best_class]])
  
    # Sort the B-boxes by their final score
    thresholded_predictions.sort(key=lambda tup: tup[1], reverse=True)
  
    # Non maximal suppression
    if (len(thresholded_predictions) > 0):
        nms_predictions = non_maximal_suppression(thresholded_predictions, 0.3)
    else:
        nms_predictions = []
    
    label_boxes = []
    # Append B-Boxes
    for i in range(len(nms_predictions)):
  
        best_class_name = nms_predictions[i][2]
        lefttop = tuple(nms_predictions[i][0][0:2])
        rightbottom = tuple(nms_predictions[i][0][2:4])
        color = colors[classes.index(nms_predictions[i][2])]

        label_boxes.append((best_class_name, lefttop, rightbottom, color))
    
    return label_boxes

def iou(boxA,boxB):
  # boxA = boxB = [x1,y1,x2,y2]

  # Determine the coordinates of the intersection rectangle
  xA = max(boxA[0], boxB[0])
  yA = max(boxA[1], boxB[1])
  xB = min(boxA[2], boxB[2])
  yB = min(boxA[3], boxB[3])

  # Compute the area of intersection
  intersection_area = (xB - xA + 1) * (yB - yA + 1)

  # Compute the area of both rectangles
  boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
  boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

  # Compute the IOU
  iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

  return iou

def non_maximal_suppression(thresholded_predictions, iou_threshold):

  nms_predictions = []

  # Add the best B-Box because it will never be deleted
  nms_predictions.append(thresholded_predictions[0])

  # For each B-Box (starting from the 2nd) check its iou with the higher score B-Boxes
  # thresholded_predictions[i][0] = [x1,y1,x2,y2]
  i = 1
  while i < len(thresholded_predictions):
    n_boxes_to_check = len(nms_predictions)
    #print('N boxes to check = {}'.format(n_boxes_to_check))
    to_delete = False

    j = 0
    while j < n_boxes_to_check:
        curr_iou = iou(thresholded_predictions[i][0],nms_predictions[j][0])
        if(curr_iou > iou_threshold ):
            to_delete = True
        #print('Checking box {} vs {}: IOU = {} , To delete = {}'.format(thresholded_predictions[i][0],nms_predictions[j][0],curr_iou,to_delete))
        j = j+1

    if to_delete == False:
        nms_predictions.append(thresholded_predictions[i])
    i = i+1

  return nms_predictions

def sigmoid(x):
    return 1 / (1 + np.e ** -x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis = 0)


