import streamlit as st
import os
import cv2
import pandas as pd
from PIL import Image
import numpy as np

# --- 1. 초기 설정 및 세션 상태 관리 ---
if 'file_idx' not in st.session_state:
    st.session_state.file_idx = 0
if 'history' not in st.session_state:
    st.session_state.history = []

st.set_page_config(layout="wide", page_title="ODcheck Pro - Enhanced")

# --- 2. 헬퍼 함수 ---
def load_yolo_labels(label_path, img_w, img_h):
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                cls, x, y, w, h = map(float, line.split())
                # 정규화 좌표를 픽셀 좌표로 변환
                x1 = int((x - w/2) * img_w)
                y1 = int((y - h/2) * img_h)
                x2 = int((x + w/2) * img_w)
                y2 = int((y + h/2) * img_h)
                labels.append({'class': int(cls), 'bbox': [x1, y1, x2, y2], 'raw': [cls, x, y, w, h]})
    return labels

def draw_bboxes(image, labels, selected_idx=None):
    img_draw = image.copy()
    for i, label in enumerate(labels):
        x1, y1, x2, y2 = label['bbox']
        color = (0, 255, 0) if i != selected_idx else (255, 0, 0) # 선택된 객체는 빨간색
        thickness = 2 if i != selected_idx else 5
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(img_draw, f"ID:{i} | Cls:{label['class']}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return img_draw

# --- 3. 사이드바: 파일 탐색 및 필터 ---
st.sidebar.title("📁 데이터셋 설정")
img_dir = st.sidebar.text_input("이미지 경로", "data/images")
label_dir = st.sidebar.text_input("라벨 경로", "data/labels")

images = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
if not images:
    st.error("이미지를 찾을 수 없습니다.")
    st.stop()

st.sidebar.markdown("---")
file_select = st.sidebar.selectbox("파일 직접 선택", images, index=st.session_state.file_idx)
st.session_state.file_idx = images.index(file_select)

# --- 4. 메인 화면 구성 ---
col1, col2 = st.columns([0.7, 0.3])

img_path = os.path.join(img_dir, images[st.session_state.file_idx])
lbl_path = os.path.join(label_dir, images[st.session_state.file_idx].rsplit('.', 1)[0] + ".txt")

image = cv2.imread(img_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
img_h, img_w, _ = image.shape
labels = load_yolo_labels(lbl_path, img_w, img_h)

with col1:
    st.subheader(f"🖼️ 검수 화면: {images[st.session_state.file_idx]}")
    
    # 객체 선택 (문제가 되는 위치를 찾기 쉽게 함)
    obj_to_highlight = st.selectbox("집중 점검할 객체 선택", 
                                    options=range(len(labels)), 
                                    format_func=lambda x: f"객체 {x} (Class: {labels[x]['class']})")
    
    canvas = draw_bboxes(image, labels, selected_idx=obj_to_highlight)
    st.image(canvas, use_column_width=True)

with col2:
    st.subheader("🔍 세부 정보 및 수정")
    
    if labels:
        target = labels[obj_to_highlight]
        # 선택된 객체 Crop 확대 시각화
        x1, y1, x2, y2 = target['bbox']
        # 여유 공간을 두고 크롭 (Padding)
        p = 20
        crop = image[max(0, y1-p):min(img_h, y2+p), max(0, x1-p):min(img_w, x2+p)]
        st.image(crop, caption=f"객체 {obj_to_highlight} 확대", use_column_width=True)
        
        # 수정 폼
        new_cls = st.number_input("클래스 수정", value=target['class'], step=1)
        if st.button("수정 사항 저장"):
            # 실제 파일 저장 로직 (필요에 따라 구현)
            target['class'] = new_cls
            st.session_state.history.append(f"{images[st.session_state.file_idx]} - 객체 {obj_to_highlight} 수정됨")
            st.success("수정 완료!")

    st.markdown("---")
    st.subheader("📜 수정 이력")
    for log in reversed(st.session_state.history[-5:]): # 최근 5개만 표시
        st.write(f"- {log}")

# --- 5. 하단 내비게이션 ---
st.markdown("---")
c1, c2, c3 = st.columns(3)
if c1.button("⬅️ 이전 이미지"):
    st.session_state.file_idx = max(0, st.session_state.file_idx - 1)
    st.rerun()
if c3.button("다음 이미지 ➡️"):
    st.session_state.file_idx = min(len(images)-1, st.session_state.file_idx + 1)
    st.rerun()
