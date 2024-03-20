from PIL import Image
from yolo_onnx.yolov8_onnx import YOLOv8

# initialize model
yolov8_detector = YOLOv8('last.onnx')

# load image
img = Image.open('images/123.jpg')

# do inference
detections = yolov8_detector(img, size=640, conf_thres=0.3, iou_thres=0.5)

# print results
print(detections)