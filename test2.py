from ultralytics import YOLO

def main():
    model = YOLO("/home/ryo32/YOLO-practice/runs/detect/train_on_device45/weights/best.pt")

    # 絶対パスで指定
    source_folder = "/home/ryo32/YOLO-practice/datasets/test/images"

    results = model.predict(
        source=source_folder,
        device=4,
        conf=0.5,
        save=True
    )

    print(results)

if __name__ == "__main__":
    main()
