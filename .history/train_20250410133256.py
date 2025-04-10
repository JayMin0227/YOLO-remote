from ultralytics import YOLO

# yolo11n.pt など事前学習済みモデルを指定
model = YOLO("yolo11n.pt")

# 学習を実行
model.train(
    data="path/to/data.yaml",  # 上記の data.yaml
    epochs=50,
    imgsz=640
)
