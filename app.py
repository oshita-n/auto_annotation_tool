import streamlit as st
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

import torch
from PIL import Image
import numpy as np
from io import BytesIO, BufferedReader
from ultralytics import YOLO
import cv2
import pandas as pd
import zipfile
import shutil
import os

# モデルとプロセッサのロード（グローバルに一度だけロード）
processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

# 複数のプロンプトからマスク画像を作成して、最終的に複数クラスのマスク画像をブレンドする
def process(input_image, prompts, colors):
    # プロンプトからマスク画像を作成
    mask_images = []
    for i, prompt in enumerate(prompts):
        # プロンプトからマスク画像を作成
        inputs = processor(text=prompt, images=input_image, return_tensors="pt", padding="max_length")
        outputs = model(**inputs)
        preds = torch.sigmoid(outputs.logits).squeeze().detach().cpu().numpy()
        preds = np.where(preds > 0.3, 1, 0).astype(np.uint8)
        preds = Image.fromarray(preds.astype(np.uint8))
        preds = np.array(preds.resize((input_image.width, input_image.height)))
        preds = preds * 255
        # チャンネルを3にする
        preds = np.repeat(preds[:, :, np.newaxis], 3, axis=2)
        # 白い部分を任意のカラーに変換
        preds[:, :, 0] = np.where(preds[:, :, 0] == 255, colors[i][0], preds[:, :, 0])
        preds[:, :, 1] = np.where(preds[:, :, 1] == 255, colors[i][1], preds[:, :, 1])
        preds[:, :, 2] = np.where(preds[:, :, 2] == 255, colors[i][2], preds[:, :, 2])
        preds = Image.fromarray(preds.astype(np.uint8))
        mask_images.append(preds)

    # 画像を重ねていくが、重なった部分は最後に重ねたクラスの色にする
    all_mask_image = Image.new("RGB", (input_image.width, input_image.height))
    all_mask_image = np.array(all_mask_image)
    for i, mask in enumerate(mask_images):
        all_mask_image = np.where(np.array(mask) != 0, np.array(mask), all_mask_image)

    all_mask_image = Image.fromarray(all_mask_image.astype(np.uint8))
    # 元の画像にマスクを重ねる
    all_mask_image = Image.blend(input_image, all_mask_image, 0.5)
    return all_mask_image

# クラスごとにカラーで領域を塗りつぶし、最終的に複数クラスのカラーを合わせる
def process_without_blend(input_image, prompts, colors):
    width, height = input_image.size
    # プロンプトからクラスマップ画像を作成
    class_maps = []
    for i, prompt in enumerate(prompts):
        # プロンプトからマスク画像を作成
        inputs = processor(text=prompt, images=input_image, return_tensors="pt", padding="max_length")
        outputs = model(**inputs)
        preds = torch.sigmoid(outputs.logits).squeeze().detach().cpu().numpy()
        preds = np.where(preds > 0.3, 1, 0).astype(np.uint8)
        preds = Image.fromarray(preds.astype(np.uint8))
        preds = np.array(preds.resize((width, height)))
        preds = preds * 255
        # チャンネルを3にする
        preds = np.repeat(np.array(preds)[:, :, np.newaxis], 3, axis=2)
        # 白い部分を任意のカラーに変換
        preds[:, :, 0] = np.where(preds[:, :, 0] == 255, colors[i][0], preds[:, :, 0])
        preds[:, :, 1] = np.where(preds[:, :, 1] == 255, colors[i][1], preds[:, :, 1])
        preds[:, :, 2] = np.where(preds[:, :, 2] == 255, colors[i][2], preds[:, :, 2])
        preds = Image.fromarray(preds.astype(np.uint8))
        class_maps.append(preds)

    # 画像を重ねていくが、重なった部分は最後に重ねたクラスの色にする
    all_class_map = Image.new("RGB", (width, height))
    all_class_map = np.array(all_class_map)
    for i, class_map in enumerate(class_maps):
        all_class_map = np.where(np.array(class_map) != 0, np.array(class_map), all_class_map)
    all_class_map = Image.fromarray(all_class_map.astype(np.uint8))
    return all_class_map


def get_image_download_binary(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img = BufferedReader(buffered)
    return img

def main():
    # Streamlitアプリのタイトル
    st.title("Auto Annotation App")

    # 画像アップロード(複数の画像のアップロードを許可)
    uploaded_files = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    # アップロードされた画像ごとにラベル入力
    if uploaded_files is not None:
        # ラベル入力
        label = st.text_input("ラベルを入力してください。複数の場合はカンマ区切りで入力してください。")

        # runボタンを設置
        segmentation_run = st.button("segmentation")
        detection_run = st.button("detection")

        # runボタンが押されたらモデルを実行
        if segmentation_run:
            # 画像を読み込み推論する
            # 最終的にzipにしてダウンロードできるようにする
            if not os.path.exists("tmp"):
                os.mkdir("tmp")
            else:
                # すでにtmpフォルダがある場合は削除して再作成
                shutil.rmtree("tmp")
                os.mkdir("tmp")
                
            if not os.path.exists("tmp/annotation"):
                os.mkdir("tmp/annotation")
            if not os.path.exists("tmp/inputs"):
                os.mkdir("tmp/inputs")
            if not os.path.exists("tmp/preds"):
                os.mkdir("tmp/preds")
            if not os.path.exists("tmp/no_annotation"):
                os.mkdir("tmp/no_annotation")

            # 検出レポートを作成する
            csv_data = "detection,no_detection\n"

            for uploaded_file in uploaded_files:
                input_image = Image.open(uploaded_file)
                labels = label.split(",")
                colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 255, 255], [255, 0, 255]]
                preds = process(input_image, labels, colors)
                anno_img = process_without_blend(input_image, labels, colors)

                input_image.save(f"tmp/inputs/{uploaded_file.name}")
                # アノテーションがない場合はno_annotationフォルダにコピー
                if np.sum(np.array(anno_img)) == 0:
                    shutil.copy(f"tmp/inputs/{uploaded_file.name}", f"tmp/no_annotation/{uploaded_file.name}")
                    csv_data += f",{uploaded_file.name}\n"
                else:
                    # アノテーションがある場合はannotationフォルダにコピー
                    anno_img.save(f"tmp/annotation/{uploaded_file.name}")
                    csv_data += f"{uploaded_file.name},\n"
                preds.save(f"tmp/preds/{uploaded_file.name}")
            # レポートを作成
            with open("tmp/report.csv", "w") as f:
                f.write(csv_data)
            shutil.make_archive("annotation", 'zip', root_dir="tmp")
            # アノテーションをダウンロードするボタンを設置
            btn = st.download_button("アノテーション画像をzipにまとめてダウンロードする", data=open("annotation.zip", "rb"), file_name="annotation.zip", mime="application/zip")
        elif detection_run:
            model = YOLO('yolov8n.pt')

            # 画像を読み込み推論する
            # 最終的にzipにしてダウンロードできるようにする
            if not os.path.exists("tmp"):
                os.mkdir("tmp")
            else:
                # すでにtmpフォルダがある場合は削除して再作成
                shutil.rmtree("tmp")
                os.mkdir("tmp")
                
            if not os.path.exists("tmp/annotation"):
                os.mkdir("tmp/annotation")
            if not os.path.exists("tmp/inputs"):
                os.mkdir("tmp/inputs")
            if not os.path.exists("tmp/preds"):
                os.mkdir("tmp/preds")
            if not os.path.exists("tmp/no_annotation"):
                os.mkdir("tmp/no_annotation")

            # 検出レポートを作成する
            csv_data = "detection,no_detection\n"
            for uploaded_file in uploaded_files:
                img = Image.open(uploaded_file)
                results = model.predict(img)
                # インデックスとクラスの辞書を取得
                class_name_dict = model.names
                
                annotation_class = [k for k, v in class_name_dict.items() if v == label][0]

                print(f"boxの中身: {results[0].boxes.xyxy}")
                detection_boxes = []
                # 対象のBBoxを抽出
                for i, cls in enumerate(results[0].boxes.cls):
                    if cls == annotation_class:
                        detection_boxes.append(results[0].boxes.xyxy[i].tolist())
                # アノテーションのcsvを作成
                # カラムはlabel, xmin, ymin, xmax, ymax
                anno_csv = pd.DataFrame(columns=["class", "xmin", "ymin", "xmax", "ymax"])
                anno_csv["class"] = [label] * len(detection_boxes)
                print(f"boxの中身: {detection_boxes}")
                if len(detection_boxes) > 0:
                    anno_csv[["xmin", "ymin", "xmax", "ymax"]] = detection_boxes
                # アノテーションのcsvを保存
                anno_csv.to_csv(f"tmp/annotation/{os.path.splitext(uploaded_file.name)[0]}.csv", index=False)

                # BBoxを描画
                if len(detection_boxes) > 0:
                    for box in detection_boxes:
                        img = cv2.rectangle(np.array(img), (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)
                    img = Image.fromarray(img)
                else:
                    img = Image.fromarray(np.array(img))
                input_image = Image.open(uploaded_file)
                input_image.save(f"tmp/inputs/{uploaded_file.name}")
                # アノテーションがない場合はno_annotationフォルダにコピー
                if len(detection_boxes) == 0:
                    shutil.copy(f"tmp/inputs/{uploaded_file.name}", f"tmp/no_annotation/{uploaded_file.name}")
                    csv_data += f",{uploaded_file.name}\n"
                else:
                    # アノテーションがある場合はannotationフォルダにコピー
                    img.save(f"tmp/annotation/{uploaded_file.name}")
                    csv_data += f"{uploaded_file.name},\n"
                img.save(f"tmp/preds/{uploaded_file.name}")
            # レポートを作成
            with open("tmp/report.csv", "w") as f:
                f.write(csv_data)
            shutil.make_archive("annotation", 'zip', root_dir="tmp")
            # アノテーションをダウンロードするボタンを設置
            btn = st.download_button("アノテーション画像をzipにまとめてダウンロードする", data=open("annotation.zip", "rb"), file_name="annotation.zip", mime="application/zip")                    
        
    # フッター
    st.markdown(
        """
        <style>
        footer {visibility: hidden;}
        </style>
        """,
        unsafe_allow_html=True
    )      

if __name__ == "__main__":
    main()

