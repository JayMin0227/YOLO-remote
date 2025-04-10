# train_device4.py

from ultralytics import YOLO

def main():
    # 事前学習済みモデル (yolo11n.pt などを指定)
    model = YOLO("yolo11n.pt")

    # 学習の実行: device=4 で 4番目のGPUを使用
    model.train(
        data="datasets/data.yaml",  # data.yaml のパスを適宜書き換え
        epochs=50,                  # 学習エポック数
        device=4,                   # ← GPU ID=4
        name="train_on_device4"     # 結果を保存するフォルダ名
    )

if __name__ == "__main__":
    main()

