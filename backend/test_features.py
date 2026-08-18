import cv2
import os
import glob
import numpy as np

def extract_features(img, net):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist_feat = hist.flatten()
    
    blob = cv2.dnn.blobFromImage(img, 1.0/255.0, (224, 224), (0.485, 0.456, 0.406), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()[0]
    preds_norm = preds / (np.linalg.norm(preds) + 1e-7)
    
    return np.concatenate([preds_norm * 0.7, hist_feat * 0.3])

def main():
    onnx_path = os.path.join("model", "mobilenetv2.onnx")
    if not os.path.exists(onnx_path):
        print("ONNX model not found")
        return
        
    net = cv2.dnn.readNetFromONNX(onnx_path)
    dataset = {}
    for path in glob.glob(os.path.join("dataset", "*.png")):
        name = os.path.basename(path).replace(".png", "")
        img = cv2.imread(path)
        if img is not None:
            dataset[name] = (img, extract_features(img, net))

    for target_name, (target_img, target_feat) in dataset.items():
        sims = {}
        for name, (img, feat) in dataset.items():
            sim = float(np.dot(target_feat, feat) / (np.linalg.norm(target_feat) * np.linalg.norm(feat)))
            sims[name] = round(sim * 100, 2)
        print(f"Query {target_name}: {sims}")

if __name__ == "__main__":
    main()
