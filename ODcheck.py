import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# --- 페이지 설정 ---
st.set_page_config(page_title="Advanced ODcheck Tool", layout="wide")

st.title("🔍 Advanced Object Detection Checker")

# --- 사이드바: 설정 및 제어 ---
# 변수 초기화 (에러 방지)
img_dir = ""
label_dir_a = ""
label_dir_b = ""

with st.sidebar:
    st.header("📂 데이터 경로 설정")
    # 기본 경로를 현재 디렉토리(".") 등으로 설정하여 초기 에러 방지
    img_dir = st.text_input("이미지 폴더 경로", value="data/images")
    label_dir_a = st.text_input("라벨 폴더 A (GT/기존)", value="data/labels")
    label_dir_b = st.text_input("라벨 폴더 B (비교용 - 선택사항)", value="")
    
    st.header("🎨 시각화 설정")
    line_thickness = st.slider("선 두께", 1, 10, 2)
    bbox_opacity = st.slider("박스 투명도", 0.0, 1.0, 0.5)
    
    st.header("🏷️ 클래스 정의")
    class_names_input = st.text_area("클래스 목록 (쉼표 구분)", "Person, Car, Bicycle, Dog")
    class_names = [c.strip() for c in class_names_input.split(",")]
    
    target_classes = st.multiselect("확인할 클래스 필터", class_names, default=class_names)

# --- 유틸리티 함수 ---
def get_color(idx):
    """클래스 ID별 고정 색상"""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    return colors[idx % len(colors)]

def load_yolo_labels(label_path, img_w, img_h):
    if not os.path.exists(label_path):
        return []
    labels = []
    try:
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.split()
                if len(parts) < 5: continue
                cls_id = int(parts[0])
                x_c, y_c, w, h = map(float, parts[1:5])
                x1 = int((x_c - w/2) * img_w)
                y1 = int((y_c - h/2) * img_h)
                x2 = int((x_c + w/2) * img_w)
                y2 = int((y_c + h/2) * img_h)
                labels.append({'id': cls_id, 'bbox': [x1, y1, x2, y2]})
    except Exception as e:
        st.error(f"라벨 로드 에러: {e}")
    return labels

def draw_boxes(image, labels, class_names, target_classes, thickness, opacity):
    overlay = image.copy()
    for label in labels:
        cls_id = label['id']
        if cls_id >= len(class_names) or class_names[cls_id] not in target_classes:
            continue
        color = get_color(cls_id)
        x1, y1, x2, y2 = label['bbox']
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        label_text = class_names[cls_id]
        cv2.putText(overlay, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)

# --- 메인 로직 ---
# 1. 경로 유효성 검사
if img_dir and os.path.exists(img_dir):
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    if not image_files:
        st.info("선택한 폴더에 이미지 파일이 없습니다. 경로를 확인해주세요.")
    else:
        # 파일 선택 및 이동
        idx = st.select_slider("이미지 선택", options=range(len(image_files)), 
                               format_func=lambda i: image_files[i])
        
        current_img_name = image_files[idx]
        img_path = os.path.join(img_dir, current_img_name)
        
        # 이미지 로드
        raw_img = cv2.imread(img_path)
        if raw_img is not None:
            img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
            h, w, _ = img.shape
            
            # 라벨 로드
            label_name = os.path.splitext(current_img_name)[0] + ".txt"
            
            # 레이아웃 구성
            if label_dir_b and os.path.exists(label_dir_b):
                col1, col2 = st.columns(2)
                with col1:
                    st.caption(f"Source A: {label_dir_a}")
                    labels_a = load_yolo_labels(os.path.join(label_dir_a, label_name), w, h)
                    res_a = draw_boxes(img.copy(), labels_a, class_names, target_classes, line_thickness, bbox_opacity)
                    st.image(res_a, use_container_width=True)
                with col2:
                    st.caption(f"Source B: {label_dir_b}")
                    labels_b = load_yolo_labels(os.path.join(label_dir_b, label_name), w, h)
                    res_b = draw_boxes(img.copy(), labels_b, class_names, target_classes, line_thickness, bbox_opacity)
                    st.image(res_b, use_container_width=True)
            else:
                st.caption(f"이미지: {current_img_name} | 라벨 폴더: {label_dir_a}")
                labels_a = load_yolo_labels(os.path.join(label_dir_a, label_name), w, h)
                res_a = draw_boxes(img.copy(), labels_a, class_names, target_classes, line_thickness, bbox_opacity)
                st.image(res_a, use_container_width=True)
        else:
            st.error("이미지를 불러올 수 없습니다.")
else:
    st.info("좌측 사이드바에서 이미지 폴더 경로를 입력해주세요.")
    st.write("현재 입력된 경로:", img_dir)
