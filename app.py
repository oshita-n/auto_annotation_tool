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

def process(input_image, prompt):
    inputs = processor(text=prompt, images=input_image, padding="max_length", return_tensors="pt")
    # predict
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.sigmoid(outputs.logits).squeeze().detach().cpu().numpy()
    preds = np.where(preds > 0.3, 1, 0).astype(np.uint8)
    preds = Image.fromarray(preds.astype(np.uint8))
    preds = np.array(preds.resize((input_image.width, input_image.height)))
    
    # 推論結果の赤色のマスクを作成
    masked_image = np.array(input_image)
    masked_image[preds==1] = [255, 0, 0]
    masked_image = Image.fromarray(masked_image)
    # 元の画像にマスクを重ねる
    masked_image = Image.blend(input_image, masked_image, 0.5)
    return masked_image

def process_without_blend(input_image, prompt):
    inputs = processor(text=prompt, images=input_image, padding="max_length", return_tensors="pt")
    # predict
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.sigmoid(outputs.logits).squeeze().detach().cpu().numpy()
    preds = np.where(preds > 0.3, 1, 0).astype(np.uint8)
    preds = Image.fromarray(preds.astype(np.uint8))
    preds = np.array(preds.resize((input_image.width, input_image.height)))
    preds = preds * 255
    preds = Image.fromarray(preds.astype(np.uint8))
    return preds

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
        label = st.text_input("ラベルを入力してください")

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
                preds = process(input_image, label)
                anno_img = process_without_blend(input_image, label)
                anno_img = Image.fromarray(np.array(anno_img))
                input_image.save(f"tmp/inputs/{uploaded_file.name}")
                # アノテーションがない場合はno_annotationフォルダにコピー
                if np.sum(np.array(anno_img)) == 0:
                    shutil.copy(f"tmp/inputs/{uploaded_file.name}", f"tmp/no_annotation/{uploaded_file.name}")
                    csv_data += f",{uploaded_file.name}\n"
                else:
                    # アノテーションがある場合はannotationフォルダにコピー
                    anno_img.save(f"tmp/annotation/{uploaded_file.name}.jpg")
                    csv_data += f"{uploaded_file.name},\n"
                preds.save(f"tmp/preds/{uploaded_file.name}.jpg")
            # レポートを作成
            with open("tmp/report.csv", "w") as f:
                f.write(csv_data)
            shutil.make_archive("annotation", 'zip', root_dir="tmp")
            # アノテーションをダウンロードするボタンを設置
            btn = st.download_button("アノテーション画像をzipにまとめてダウンロードする", data=open("annotation.zip", "rb"), file_name="annotation.zip", mime="application/zip")
        elif detection_run:
            model = YOLO('yolov8n.pt')
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
            anno_csv[["xmin", "ymin", "xmax", "ymax"]] = detection_boxes
            # BBoxを描画
            if len(detection_boxes) > 0:
                for box in detection_boxes:
                    img = cv2.rectangle(np.array(img), (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 3)
                img = Image.fromarray(img)
                st.image(img, caption="予測結果")
                # アノテーションのcsvをダウンロードするボタンを設置
                btn = st.download_button("アノテーションをダウンロードする", data=anno_csv.to_csv(index=False), file_name=f"{uploaded_file.name}_annotation.csv", mime="text/csv")
            else:
                st.write("ラベルが見つかりませんでした。")            


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

