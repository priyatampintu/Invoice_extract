from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO()

model.train(data="data.yaml", epochs = 200, imgsz=800, plots=True)
