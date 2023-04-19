import numpy as np
import cv2
import time

def memory_usage():
    """Memory usage of the current process in kilobytes."""
    status = None
    result = {'peak': 0, 'rss': 0}
    try:
        # This will only work on systems with a /proc file system
        # (like Linux).
        status = open('/proc/self/status')
        for line in status:
            parts = line.split()
            key = parts[0][2:-1].lower()
            if key in result:
                result[key] = int(parts[1])
    finally:
        if status is not None:
            status.close()
    return result





# # # # #   This is for obj. detection - START   # # # # #
RESIZED_DIMENSIONS = (300, 300)  # Dimensions that SSD was trained on.
IMG_NORM_RATIO = 0.007843  # In grayscale a pixel can range between 0 and 255

# Load the pre-trained neural network
neural_network = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt.txt',
                                          'MobileNetSSD_deploy.caffemodel')

# List of categories and classes
categories = {0: 'background', 1: 'aeroplane', 2: 'bicycle', 3: 'bird',
              4: 'boat', 5: 'bottle', 6: 'bus', 7: 'car', 8: 'cat',
              9: 'chair', 10: 'cow', 11: 'diningtable', 12: 'dog',
              13: 'horse', 14: 'motorbike', 15: 'person',
              16: 'pottedplant', 17: 'sheep', 18: 'sofa',
              19: 'train', 20: 'tvmonitor'}

classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person",
           "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Create the bounding boxes
bbox_colors = np.random.uniform(255, 0, size=(len(categories), 3))

# # # # #   This is for obj. detection - END   # # # # #






# Find OpenCV version
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

# Start camera capture
cap = cv2.VideoCapture("http://192.168.0.20:8081")

# Testing the camera
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# # # # #   Get FPS with OpenCV
if int(major_ver)  < 3 :
    fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
else :
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


# # # # #   Get FPS with "manual" calculation
# Number of frames to capture
num_frames = 60;

print("Capturing {0} frames for detecting real FPS".format(num_frames))

# Start time
start = time.time()

# Grab a few frames
for i in range(0, num_frames) :
    ret, frame = cap.read()

# End time
end = time.time()

# Time elapsed
seconds = end - start
print ("Time taken : {0} seconds".format(seconds))

# Calculate frames per second
fps = num_frames / seconds
fps = int(fps)
fps_string = str(fps)
print("Estimated frames per second : {0}".format(fps))



# Creating frame buffer

frame_buffer_in_sec = 3
frame_buffer_sum_frames = fps * frame_buffer_in_sec
frame_buffer = []
frame_counter = 0

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    # print(memory_usage())
    # deleting the oldest frame from the buffer
    if  len(frame_buffer) >= frame_buffer_sum_frames :
       del frame_buffer[0]

    # Reseting frame_counter
    if frame_counter >= fps :
        frame_counter = 0
    
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Capture the frame's height and width
    (h, w) = frame.shape[:2]

    # Drawing the FPS to the image
    # cv2.putText(frame, fps_string, (7, 70), font, 3, (100, 255, 0), 3, cv2.LINE_AA)

    # copying the actual frame to the frame_buffer
    frame_buffer.append(frame)


    # Checkging one frame in every sec with obj. detection (mobile ssd)
    if frame_counter == 0 :
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        ct_int = int(time.strftime("%H%M%S", t))
        print(current_time)
        st = time.time()
        
        # Create a blob. A blob is a group of connected pixels in a binary
        # frame that share some common property (e.g. grayscale value)
        # Preprocess the frame to prepare it for deep learning classification
        frame_blob = cv2.dnn.blobFromImage(cv2.resize(frame, RESIZED_DIMENSIONS),
                     IMG_NORM_RATIO, RESIZED_DIMENSIONS, 127.5)

        # Set the input for the neural network
        neural_network.setInput(frame_blob)

        # Predict the objects in the image
        neural_network_output = neural_network.forward()
        print(len(neural_network_output))
        # Put the bounding boxes around the detected objects
        for i in np.arange(0, neural_network_output.shape[2]):
            confidence = neural_network_output[0, 0, i, 2]

            # Confidence must be at least 30%
            if confidence > 0.30:
                idx = int(neural_network_output[0, 0, i, 1])

                bounding_box = neural_network_output[0, 0, i, 3:7] * np.array(
                    [w, h, w, h])

                (startX, startY, endX, endY) = bounding_box.astype("int")

                label = "{}: {:.2f}%".format(classes[idx], confidence * 100)

                cv2.rectangle(frame, (startX, startY), (
                    endX, endY), bbox_colors[idx], 2)

                y = startY - 15 if startY - 15 > 15 else startY + 15

                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, bbox_colors[idx], 2)
                print("Found something! Writing jpg...")
                cv2.imwrite("frame%d.jpg" % ct_int, frame)

        et = time.time()
        # get the obj. det. execution time
        elapsed_time = et - st
        print('Execution time:', elapsed_time, 'seconds')


    
    # Display the resulting frame
    cv2.imshow('frame', frame)

    frame_counter = frame_counter +1

    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
print("OpenCV major, minor, sub versions: " , major_ver,".",minor_ver,".",subminor_ver)
print(len(frame_buffer))
