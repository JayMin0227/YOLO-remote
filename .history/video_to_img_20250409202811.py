import cv2
import os

# 動画ファイルのパスを指定
video_path = "20241216_0255_Primal Rain Ritual_simple_compose_01jf5qc3tzfd2scvyfnc3f2g4h.mp4"
output_dir = "output_images"

# 出力先ディレクトリが存在しない場合は作成
os.makedirs(output_dir, exist_ok=True)

# 動画ファイルを読み込み
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("動画ファイルを開けませんでした。パスを確認してください。")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    
    # フレームの読み取りが終了したらループを抜ける
    if not ret:
        print("動画の再生が終了しました。")
        break
    
    # フレームを表示
    cv2.imshow('Video Frame', frame)
    
    # キー入力を待つ
    key = cv2.waitKey(1) & 0xFF
    
    # スペースキーを押した場合
    if key == ord(' '):
        # フレームを画像として保存
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        print(f"フレームを保存しました: {frame_filename}")
    
    # 'q'キーで終了
    if key == ord('q'):
        print("終了します。")
        break
    
    frame_count += 1

# リソースを解放
cap.release()
cv2.destroyAllWindows()

