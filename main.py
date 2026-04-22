import ultralytics
import supervision
import torch
import cv2
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def main():
    model = YOLO("yolov8n.pt")
    results = model("https://ultralytics.com/images/bus.jpg")

    for result in results:
        result.show()
        for box in result.boxes:
            print(result.names[int(box.cls[0])])
            print(box.xyxy)
    fig, ax, = plt.subplot()
 
if __name__ == "__main__":
    main()