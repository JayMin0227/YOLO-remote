# train_device4.py

from ultralytics import YOLO

def main():
    
    model = YOLO("yolo11n.pt")

    # 学習の実行: device=4 で 4番目のGPUを使用
    model.train(
        data="datasets/data.yaml",  
        epochs=50,   
        device=4,  
        name="train_on_device4"     
    )

if __name__ == "__main__":
    main()

