import os
import requests

def main():
    url = "http://localhost:8000/predict"
    images = ["nilgiri_tahr.png", "ibex.png", "marco_polo.png", "red_panda.png"]
    for img_name in images:
        path = os.path.join("dataset", img_name)
        if not os.path.exists(path):
            print(f"Missing {path}")
            continue
        with open(path, "rb") as f:
            res = requests.post(url, files={"image": (img_name, f, "image/png")})
            data = res.json()
            print(f"Image: {img_name:<16} -> Species: {data.get('species'):<15} (Confidence: {data.get('confidence')}%)")

if __name__ == "__main__":
    main()
