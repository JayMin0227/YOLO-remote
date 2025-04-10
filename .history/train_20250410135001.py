from ultralytics import YOLO

# yolo11n.pt など事前学習済みモデルを指定
model = YOLO("yolo11n.pt")

# 学習を実行
model.train(
    data="datasets/images/People_Detection.v8i.yolov11/data.yaml",
    epochs=50
)

