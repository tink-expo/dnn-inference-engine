import sys
import os
import numpy as np
import cv2 as cv2
import time
import yolov2tiny

# Constants.
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

def save_tensors(tensors):
    save_dir = os.path.join(os.getcwd(), "intermediate")
    os.makedirs(save_dir, exist_ok=True)
    for i in range(len(tensors)):
        save_path = os.path.join(save_dir, "layer_{}.npy".format(i))
        np.save(save_path, tensors[i])

def resize_input(im):
    imsz = cv2.resize(im, (k_input_height, k_input_width))
    imsz = imsz / 255.
    imsz = imsz[:,:,::-1]
    return np.asarray(imsz, dtype=np.float32)

def draw_output_frame(frame, label_boxes):
    def re_resize_pos(pos):
        return (
            int(pos[0] * frame.shape[1] / k_input_width),
            int(pos[1] * frame.shape[0] / k_input_height))

    for lb in label_boxes:
        best_class_name, lefttop, rightbottom, color = lb
        print(lefttop, rightbottom)
        lefttop = re_resize_pos(lefttop)
        rightbottom = re_resize_pos(rightbottom)
        frame = cv2.rectangle(frame, lefttop, rightbottom, color)
        text_pos = (
            int((lefttop[0] + rightbottom[0]) / 2),
            int((lefttop[1] + rightbottom[1]) / 2))
        cv2.putText(
            frame, best_class_name, text_pos, 
            cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, 
            color=color, thickness=1)
    return frame

def video_object_detection(in_video_path, out_video_path, proc="cpu"):
    #
    # This function runs the inference for each frame and creates the output video.
    #

    in_video, out_video = open_video_with_opencv(in_video_path, out_video_path)

    # Create an instance of the YOLO_V2_TINY class. Pass the dimension of
    # the input, a path to weight file, and which device you will use as arguments.

    weight_pickle_path = os.path.join(os.getcwd(), '../test-proj3/y2t_weights.pickle')
    model = yolov2tiny.YOLO_V2_TINY(
            [1, k_input_height, k_input_width, 3], weight_pickle_path, proc)

    # Start the main loop. For each frame of the video, the loop must do the followings:
    # 1. Do the inference.
    # 2. Run postprocessing using the inference result, accumulate them through the video writer object.
    #    The coordinates from postprocessing are calculated according to resized input; you must adjust
    #    them to fit into the original video.
    # 3. Measure the end-to-end time and the time spent only for inferencing.
    # 4. Save the intermediate values for the first frame.
    # Note that your input must be adjusted to fit into the algorithm,
    # including resizing the frame and changing the dimension.

    e2e_time = 0
    inference_time = 0
    frame_count = 0
    
    while True:
        e2e_time_start = time.time()

        ret, frame = in_video.read()
        if not ret:
            break

        frame_count += 1

        input_img = resize_input(frame)
        input_img = np.expand_dims(input_img, 0)

        inference_time_start = time.time()
        predictions = model.inference(input_img)
        inference_time += time.time() - inference_time_start

        label_boxes = yolov2tiny.postprocessing(predictions[-1])
        
        frame = draw_output_frame(frame, label_boxes)
        out_video.write(frame)

        e2e_time += time.time() - e2e_time_start

        # Exclude time for save_tensors in e2e time.
        if frame_count == 1:
            save_tensors(predictions)

    # Check the inference peformance; end-to-end elapsed time and inferencing time.
    # Check how many frames are processed per second respectivly.
    inference_fps = frame_count / inference_time
    e2e_fps = frame_count / e2e_time
    print("Inference time: {}".format(inference_time))
    print("End-to-end time: {}".format(e2e_time))
    print("Inference fps: {}".format(inference_fps))
    print("End-to-end fps: {}".format(e2e_fps))

    # Release the opened videos.
    in_video.release()
    out_video.release()

#
# Extra functions for testing and debugging.
#

# To check if last layer for first frame is saved properly.
def photo_write(in_video_path, out_photo_path, tensor_path='./intermediate/layer_39.npy'):
    in_video = cv2.VideoCapture(in_video_path)
    ret, frame = in_video.read()
    prediction = np.load(tensor_path)
    label_boxes = yolov2tiny.postprocessing(prediction)
    frame = draw_output_frame(frame, label_boxes)
    cv2.imwrite(out_photo_path, frame)
    in_video.release()

# Single photo detection for debugging.
def photo_object_detection(in_photo_path, out_photo_path, proc="cpu"):
    frame = cv2.imread(in_photo_path)

    weight_pickle_path = os.path.join(os.getcwd(), '../test-proj3/y2t_weights.pickle')
    model = yolov2tiny.YOLO_V2_TINY(
            [1, k_input_height, k_input_width, 3], weight_pickle_path, proc)
    
    input_img = resize_input(frame)
    input_img = np.expand_dims(input_img, 0)

    predictions = model.inference(input_img)
    save_tensors(predictions)

    label_boxes = yolov2tiny.postprocessing(predictions[-1])
    
    frame = draw_output_frame(frame, label_boxes)
    cv2.imwrite(out_photo_path, frame)

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

    photo_object_detection(in_video_path, out_video_path, proc)

if __name__ == "__main__":
    main()
