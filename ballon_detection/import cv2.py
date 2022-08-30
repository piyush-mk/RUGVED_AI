import cv2
import sys

(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
 
if __name__ == '__main__' :
    tracker_type = 'MIL'
 
    if int(minor_ver) < 3:
        tracker = cv2.TrackerMIL_create()
video = cv2.VideoCapture("vid.mp4")
if not video.isOpened():
  print("Could not open video")
  sys.exit()
ok, frame = video.read()
if not ok:
  print ('Cannot read video file')
  sys.exit()