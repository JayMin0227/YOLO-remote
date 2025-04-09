from ultralytics import YOLO

model = YOLO("yolo11n.pt")

# 画像のリストを用意
images = ["output_images/bus.jpg", "output_images/terrain3.jpg"]

# まとめて推論
results_list = model.predict(images, save=True, imgsz=320, conf=0.5)
