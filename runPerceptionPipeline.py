# import packages
import cv2
import numpy as np
from urllib.request import urlopen
import os
import datetime
import time
import sys
from time import sleep
from threading import Thread, Lock
from mtcnn.mtcnn import MTCNN
import torch
import time
from DETR.utils import *
import dlib
from imutils import face_utils
from collections import OrderedDict
import matplotlib.pyplot as plt

# URL's
url_cam1 = 'http://10.0.0.250:81/stream'
url_cam2 = 'http://10.0.0.251:81/stream'
urls = [url_cam1]

# weights
dlib_weights = 'weights/shape_predictor_68_face_landmarks.dat'

# image frames
frame_camn = [None]*len(urls)

# buffer size and stream
CAMERA_BUFFRER_SIZE=4096

# mutex
mutex = Lock()

# thread for grabbing frames
def thread_grab_cam(url):
    stream = urlopen(url)
    bts = b''
    global frame_camn
    while True:
        # get stream
        bts += stream.read(CAMERA_BUFFRER_SIZE)
        jpghead = bts.find(b'\xff\xd8')
        jpgend = bts.find(b'\xff\xd9')
        # identify end of stream
        if jpghead>-1 and jpgend>-1:
            # extract image frame
            jpg = bts[jpghead:jpgend+2]
            bts = bts[jpgend+2:]
            frame_ = cv2.imdecode(np.frombuffer(jpg,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
            idx_ = url_cam1.index(url)
            mutex.acquire()
            frame_camn[idx_] = frame_
            mutex.release()
            sleep(0.1)

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions
FACIAL_LANDMARKS_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (42, 48)),
	("left_eye", (36, 42)),
	("nose", (27, 35)),
	("jaw", (0, 17))
])

# function to visualize facial landmarks
def visualize_facial_landmarks(image, shape, colors=None, alpha=0.25):
	# create two copies of the input image -- one for the
	# overlay and one for the final output image
	overlay = image.copy()
	output = image.copy()
	# if the colors list is None, initialize it with a unique
	# color for each facial landmark region
	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (79, 76, 240),
			(168, 100, 168), (168, 100, 168),
			(163, 38, 32), (180, 42, 220)]

    # loop over the facial landmark regions individually
	for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
		# grab the (x, y)-coordinates associated with the
		# face landmark
		(j, k) = FACIAL_LANDMARKS_IDXS[name]
		pts = shape[j:k]
		# check if are supposed to draw the jawline
		if name == "jaw":
			# since the jawline is a non-enclosed facial region,
			# just draw lines between the (x, y)-coordinates
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv2.line(overlay, ptA, ptB, colors[i], 2)
		# otherwise, compute the convex hull of the facial
		# landmark coordinates points and display it
		else:
			hull = cv2.convexHull(pts)
			cv2.drawContours(overlay, [hull], -1, colors[i], -1)

    # apply the transparent overlay
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
	# return the output image
	return output

# this function returns list of objects in scene as text
def obj_list_str(obj_list):
    str_ret = '| List of Objects |\n'
    for obj_ in obj_list:
        str_ret += ' - ' + obj_ + '\n'
    return str_ret

# write multi-line text on image
def multi_line_Text(image, position, text, color=(255,255,255), fontScale = 0.4, thickness = 1):
    font                   = cv2.FONT_HERSHEY_DUPLEX    
    bottomLeftCornerOfText = position
    fontColor              = color
    lineType               = 2

    dy = 15
    for i, line in enumerate(text.split('\n')):
        y = bottomLeftCornerOfText[1] + i*dy
        cv2.putText(image,line, 
            (bottomLeftCornerOfText[0], y), 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)

if __name__ == '__main__':
    threads = list()

    # Get DETR model from PyTorch hub and load it into the GPU
    detr_model = torch.hub.load('facebookresearch/detr', 'detr_resnet101', pretrained=True)
    detr_model.eval()
    detr_model = detr_model.cuda()

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_weights)
    
    # start threads
    for i in range(len(urls)):
        print('Starting capture thread for camera {}'.format(i))
        x = Thread(target=thread_grab_cam, args=(urls[i],))
        threads.append(x)
        x.start()

    # loop forever
    while True:
        try:
            for i in range(len(urls)):
                if frame_camn[i] is not None:
                    mutex.acquire()
                    grabbed_frame = frame_camn[i]
                    frame_camn[i] = None
                    mutex.release()

                    # make a copy of image for visualization
                    img_viz = grabbed_frame.copy()

                    # grid of faces
                    panel_width = 128
                    grid_panel = np.zeros(shape=(img_viz.shape[0],panel_width, img_viz.shape[2]), dtype=np.uint8)
                    grid_panel_fill = None
                    # info panel
                    info_panel = np.zeros(shape=(img_viz.shape[0],panel_width, img_viz.shape[2]), dtype=np.uint8)

                    # convert to tensor and load to GPU
                    img_tensor = transform(grabbed_frame).unsqueeze(0).cuda()

                    # perform inference
                    pred = None
                    obj_list = []
                    with torch.no_grad():
                        # forward pass through model
                        pred = detr_model(img_tensor)
                        
                    # Extract class probability and bounding-box
                    pred_logits = pred['pred_logits'][0]
                    pred_boxes = pred['pred_boxes'][0]
                    
                    # get softmax of logits and choose the index with highest probability
                    pred_prob = pred_logits.softmax(-1)
                    pred_prob_np = pred_prob.cpu().numpy()
                    pred_idx = np.argmax(pred_prob_np, -1)
                    
                    ## Filter out detections and draw bounding-box
                    # iterate through predictions
                    for j, (idx, box) in enumerate(zip(pred_idx, pred_boxes)):
                        if (idx >= len(CLASSES)) or (pred_prob_np[j][idx] < conf_thres):
                            continue

                        obj_list.append(CLASSES[idx])

                    # print objects found in the scene
                    if (len(obj_list) > 0):
                        img_viz = draw_bbox(img_viz, pred)

                        # face detection
                        rects = detector(grabbed_frame)

                        # loop over the face detections
                        for j in range(len(rects)):
                            bbox = [rects[j].left(), rects[j].top(), rects[j].right(), rects[j].bottom()]
                        
                            # determine the facial landmarks for the face region, then
                            # convert the facial landmark (x, y)-coordinates to a NumPy
                            # array
                            shape = predictor(grabbed_frame, rects[j])
                            shape = face_utils.shape_to_np(shape)
                            img_viz = visualize_facial_landmarks(img_viz, shape)

                            # loop over the facial landmark regions individually
                            leftEyePts, rightEyePts = None, None
                            for (k, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
                                # grab the (x, y)-coordinates associated with the
                                # face landmark
                                (m, n) = FACIAL_LANDMARKS_IDXS[name]
                                pts = shape[m:n]
                                # check if are supposed to draw the jawline
                                if name == "left_eye":
                                    leftEyePts = pts
                                elif name == "right_eye":
                                    rightEyePts = pts

                            '''
                            Let's find and angle of the face. First calculate 
                            the center of left and right eye by using eye landmarks.
                            '''

                            leftEyeCenter = np.array(leftEyePts).mean(axis=0).astype("int")
                            rightEyeCenter = np.array(rightEyePts).mean(axis=0).astype("int")
                            leftEyeCenter = (leftEyeCenter[0],leftEyeCenter[1])
                            rightEyeCenter = (rightEyeCenter[0],rightEyeCenter[1])

                            # draw the circle at centers and line connecting to them
                            cv2.circle(img_viz, leftEyeCenter, 2, (0, 0, 255), 5)
                            cv2.circle(img_viz, rightEyeCenter, 2, (0, 0, 255), 5)
                            cv2.line(img_viz, leftEyeCenter, rightEyeCenter, (255,255,255), 3)

                            # find and angle of line by using slop of the line.
                            dY = rightEyeCenter[1] - leftEyeCenter[1]
                            dX = rightEyeCenter[0] - leftEyeCenter[0]
                            angle = np.degrees(np.arctan2(dY, dX))

                            # to get the face at the center of the image,
                            # set desired left eye location. Right eye location 
                            # will be found out by using left eye location.
                            # this location is in percentage.
                            desiredLeftEye=(0.35, 0.35)
                            # Set the croped image(face) size after rotaion.
                            desiredFaceWidth = panel_width
                            desiredFaceHeight = panel_width

                            desiredRightEyeX = 1.0 - desiredLeftEye[0]
                            
                            # determine the scale of the new resulting image by taking
                            # the ratio of the distance between eyes in the *current*
                            # image to the ratio of distance between eyes in the
                            # *desired* image
                            dist = np.sqrt((dX ** 2) + (dY ** 2))
                            desiredDist = (desiredRightEyeX - desiredLeftEye[0])
                            desiredDist *= desiredFaceWidth
                            scale = desiredDist / dist

                            # compute center (x, y)-coordinates (i.e., the median point)
                            # between the two eyes in the input image
                            eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2,
                                (leftEyeCenter[1] + rightEyeCenter[1]) // 2)

                            # grab the rotation matrix for rotating and scaling the face
                            M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)

                            # update the translation component of the matrix
                            tX = desiredFaceWidth * 0.5
                            tY = desiredFaceHeight * desiredLeftEye[1]
                            M[0, 2] += (tX - eyesCenter[0])
                            M[1, 2] += (tY - eyesCenter[1])

                            # apply the affine transformation                                
                            cropped_face = cv2.warpAffine(grabbed_frame, M, (desiredFaceWidth, desiredFaceHeight),
                                flags=cv2.INTER_CUBIC)

                            # get boundary around face
                            margin = 16
                            face_window = np.zeros(shape=(cropped_face.shape[0]+2*margin, cropped_face.shape[1]+2*margin, cropped_face.shape[2]))
                            face_window[margin:-margin, margin:-margin] = cropped_face
                            face_window = cv2.resize(face_window, (panel_width, panel_width))

                            # Name of person
                            cv2.putText(face_window, 'person {}'.format(j), 
                                (25, int(margin/2)), 
                                cv2.FONT_HERSHEY_COMPLEX, 
                                0.4,
                                color=(255,255,255),
                                thickness=1,
                                lineType=2)

                            # resize
                            if np.all(grid_panel_fill != None):
                                grid_panel_fill = cv2.vconcat([grid_panel_fill, face_window])
                            else:
                                grid_panel_fill = face_window

                    # fit into grid panel
                    if grid_panel_fill is not None:
                        if grid_panel_fill.shape[0] < grid_panel.shape[0]:
                            grid_panel[:grid_panel_fill.shape[0]] = grid_panel_fill
                        else:
                            grid_panel = cv2.resize(grid_panel_fill, grid_panel.shape)

                    # info panel
                    multi_line_Text(info_panel, (7, 20), obj_list_str(obj_list), fontScale=0.4, thickness=1)
                    img_viz = cv2.hconcat([img_viz, grid_panel, info_panel])

                    # draw demarkation line
                    cv2.line(img_viz, (grabbed_frame.shape[1],0), (grabbed_frame.shape[1],grabbed_frame.shape[0]-1), color=(255,255,255), thickness=2)
                    cv2.line(img_viz, (grabbed_frame.shape[1]+panel_width,0), (grabbed_frame.shape[1]+panel_width,grabbed_frame.shape[0]-1), color=(255,255,255), thickness=2)

                    cv2.imshow("Camera {} Stream".format(i), img_viz)

            # wait for key
            key = cv2.waitKey(1)

        except KeyboardInterrupt:
            # close all threads
            for i, thread in enumerate(threads):
                print('Closing capture thread for camera {}'.format(i))
                thread.join()
            # break out of loop
            break
        except:
            pass

    # destroy window
    cv2.destroyAllWindows()