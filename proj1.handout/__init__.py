import sys
import os
import numpy as np
import cv2 as cv2
import time
import yolov2tiny

k_input_height = 416
k_input_width = 416

def open_video_with_opencv(in_video_path, out_video_path):
    #
    # This function takes input and output video path and open them.
    #

    # Open an object of input video using cv2.VideoCapture.
    in_video = cv2.VideoCapture(in_video_path)
    if not in_video.isOpened():
        print("Failed to open in_video {}".format(in_video_path))
        sys.exit()
    else:
        width = int(in_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(in_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = in_video.get(cv2.CAP_PROP_FPS)

    # Open an object of output video using cv2.VideoWriter.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    if not out_video.isOpened():
        print("Failed to open out_video {}".format(out_video_path))
        sys.exit()

    # Return the video objects and anything you want for further process.
    return in_video, out_video


def resize_input(im):
    imsz = cv2.resize(im, (k_input_height, k_input_width))
    imsz = imsz / 255.
    imsz = imsz[:,:,::-1]
    return np.asarray(imsz, dtype=np.float32)

def video_object_detection(in_video_path, out_video_path, proc="cpu"):
    #
    # This function runs the inference for each frame and creates the output video.
    #

    in_video, out_video = open_video_with_opencv(in_video_path, out_video_path)

    # Create an instance of the YOLO_V2_TINY class. Pass the dimension of
    # the input, a path to weight file, and which device you will use as arguments.

    weight_pickle_path = os.path.join(os.getcwd(), 'y2t_weights.pickle')
    model = yolov2tiny.YOLO_V2_TINY(
            [1, k_input_height, k_input_width, 3], weight_pickle_path, proc)
    # Start the main loop. For each frame of the video, the loop must do the followings:
    # 1. Do the inference.
    # 2. Run postprocessing using the inference result, accumulate them through the video writer object.
    #    The coordinates from postprocessing are calculated according to resized input; you must adjust
    #    them to fit into the original video.
    # 3. Measure the end-to-end time and the time spent only for inferencing.
    # 4. Save the intermediate values for the first layer.
    # Note that your input must be adjusted to fit into the algorithm,
    # including resizing the frame and changing the dimension.
    
    original_shape = None
    e2e_time = time.time()
    inference_time = 0
    
    while True:
        ret, frame = in_video.read()
        if not ret:
            break

        if original_shape is None:
            original_shape = frame.shape

        input_img = resize_input(frame)
        input_img = np.expand_dims(input_img, 0)

        inference_time_start = time.time()
        predictions = model.inference(input_img)
        inference_time += time.time() - inference_time_start

        label_boxes = yolov2tiny.postprocessing(predictions[-1])
        break

    e2e_time = time.time() - e2e_time

    # Check the inference peformance; end-to-end elapsed time and inferencing time.
    # Check how many frames are processed per second respectivly.
    

    # Release the opened videos.
    in_video.release()
    out_video.release()
    

def main():
    if len(sys.argv) < 3:
        print ("Usage: python3 __init__.py [in_video.mp4] [out_video.mp4] ([cpu|gpu])")
        sys.exit()

    in_video_path = sys.argv[1] 
    out_video_path = sys.argv[2] 

    if len(sys.argv) == 4:
        proc = sys.argv[3]
    else:
        proc = "cpu"

    video_object_detection(in_video_path, out_video_path, proc)

if __name__ == "__main__":
    main()
