import streamlit as st
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

import torch
from PIL import Image
import numpy as np
from io import BytesIO, BufferedReader

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

    # 画像アップロード
    uploaded_file = st.file_uploader("画像をアップロード", type=["png", "jpg", "jpeg"])
    # プレビューを小さく表示
    if uploaded_file:
        st.image(uploaded_file, width=200)

    # アップロードされた画像ごとにラベル入力
    if uploaded_file:
        # ラベル入力
        label = st.text_input(f"{uploaded_file.name}のラベル", key=uploaded_file)

        if label != "":
            # ラベル表示（CSSを使用してデザインをカスタマイズ）
            st.markdown(f"<div style='background-color:#4CAF50; color:white; padding:10px;'>{label}</div>", unsafe_allow_html=True)

        # runボタンを設置
        run = st.button("Run")

        # runボタンが押されたらモデルを実行
        if run:
            input_image = Image.open(uploaded_file)
            preds = process(input_image, label)
            anno_img = process_without_blend(input_image, label)
            st.image(preds, caption="予測結果")

            # アノテーションをダウンロードするボタンを設置
            btn = st.download_button("アノテーション画像をダウンロードする", data=get_image_download_binary(anno_img), file_name=f"{uploaded_file.name}_annotation.jpg", mime="image/png")
                
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
