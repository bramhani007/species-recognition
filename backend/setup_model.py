import os
import sys
import json
import urllib.request
import cv2
import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(MODEL_DIR, exist_ok=True)

ONNX_MODEL_PATH = os.path.join(MODEL_DIR, "mobilenetv2.onnx")
LABELS_PATH = os.path.join(MODEL_DIR, "imagenet_class_index.json")

ONNX_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-7.onnx"
LABELS_URL = "https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json"

def download_file(url, path, name):
    if not os.path.exists(path):
        print(f"Downloading {name}...")
        urllib.request.urlretrieve(url, path)
        print(f"Downloaded {name} ({os.path.getsize(path)} bytes).")

def main():
    download_file(LABELS_URL, LABELS_PATH, "ImageNet Labels")
    download_file(ONNX_URL, ONNX_MODEL_PATH, "MobileNetV2 ONNX Model")
    
    # Test loading model with OpenCV DNN
    print("Loading ONNX model into OpenCV DNN...")
    net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
    print("OpenCV DNN loaded model successfully!")

if __name__ == "__main__":
    main()
