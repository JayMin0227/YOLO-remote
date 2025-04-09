from ultralytics import YOLO

# モデルファイルを指定 (yolo11n.pt など)
model = YOLO("yolo11n.pt")

# 画像ファイル (bus.jpg) に対して物体検出を実行
results = model.predict("output_images/bus.jpg", save=True, imgsz=320, conf=0.5)


# これで runs/detect/predict フォルダに推論結果が保存される
