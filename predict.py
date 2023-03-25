from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("weights/best.pt")

source1='test.jpg'
im1 = Image.open(source1)
results = model.predict(source=source1,conf=0.25,save=True, save_txt=True)  # save predictions as labels
