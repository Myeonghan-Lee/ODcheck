import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# --- 페이지 설정 ---
st.set_page_config(page_title="Advanced ODcheck Tool", layout="wide")

st.title("🔍 Advanced Object Detection Checker")
st.markdown("""
이 도구는 데이터셋의 라벨 상태를 점검하고 비교하기 위해 설계되었습니다. 
좌측 사이드바에서 경로를 설정하고 필터를 적용하세요.
""")

# --- 사이드바: 설정 및 제어 ---
with st.sidebar:
    st.header("📂 데이터 경로 설정")
    img_dir = st.text_input("이미지 폴더 경로", value="./data/images")
    label_dir_a = st.text_input("라벨 폴더 A (GT/기존)", value="./data/labels_v1")
    label_dir_b = st.text_input("라벨 폴더 B (Pred/수정후 - 선택사항)", value="")
    
    st.header("🎨 시각화 설정")
    line_thickness = st.slider("선 두께", 1, 10, 2)
    bbox_opacity = st.slider("박스 투명도", 0.0, 1.0, 0.5)
    
    st.header("🏷️ 클래스 정의")
    class_names_input = st.text_area("클래스 목록 (쉼표 구분)", "Person, Car, Bicycle, Dog")
    class_names = [c.strip() for c in class_names_input.split(",")]
    
    target_classes = st.multiselect("확인할 클래스 필터", class_names, default=class_names)

# --- 유틸리티 함수 ---
def load_yolo_labels(label_path, img_w, img_h):
    if not os.path.exists(label_path):
        return []
    labels = []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            parts = line.split()
            cls_id = int(parts[0])
            # YOLO: class, x_center, y_center, width, height (normalized)
            x_c, y_c, w, h = map(float, parts[1:])
            
            # Convert to pixel coordinates
            x1 = int((x_c - w/2) * img_w)
            y1 = int((y_c - h/2) * img_h)
            x2 = int((x_c + w/2) * img_w)
            y2 = int((y_c + h/2) * img_h)
            labels.append({'id': cls_id, 'bbox': [x1, y1, x2, y2]})
    return labels

def draw_boxes(image, labels, class_names, target_classes, thickness, opacity):
    overlay = image.copy()
    # 클래스별 색상 생성
    colors = plt.cm.get_cmap('tab10', len(class_names))
    
    for label in labels:
        cls_id = label['id']
        if cls_id >= len(class_names) or class_names[cls_id] not in target_classes:
            continue
            
        color = [int(c * 255) for c in colors(cls_id)[:3]]
        x1, y1, x2, y2 = label['bbox']
        
        # 박스 그리기
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        # 라벨 텍스트
        label_text = class_names[cls_id]
        cv2.putText(overlay, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)

# --- 메인 로직 ---
if os.path.exists(img_dir):
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    image_files.sort()

    if not image_files:
        st.error("이미지 파일이 없습니다.")
    else:
        # 파일 선택 슬라이더 및 네비게이션
        col_nav1, col_nav2, col_nav3 = st.columns([1, 4, 1])
        with col_nav2:
            idx = st.select_slider("이미지 선택", options=range(len(image_files)), format_func=lambda i: image_files[i])
        
        current_img_name = image_files[idx]
        img_path = os.path.join(img_dir, current_img_name)
        
        # 이미지 로드
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # 라벨 로드
        label_name = os.path.splitext(current_img_name)[0] + ".txt"
        labels_a = load_yolo_labels(os.path.join(label_dir_a, label_name), w, h)
        
        # 화면 분할 레이아웃
        if label_dir_b:
            col1, col2 = st.columns(2)
            labels_b = load_yolo_labels(os.path.join(label_dir_b, label_name), w, h)
            
            with col1:
                st.subheader("버전 A (기존/GT)")
                res_a = draw_boxes(img.copy(), labels_a, class_names, target_classes, line_thickness, bbox_opacity)
                st.image(res_a, use_column_width=True)
                st.write(f"객체 수: {len(labels_a)}")
                
            with col2:
                st.subheader("버전 B (비교군/Pred)")
                res_b = draw_boxes(img.copy(), labels_b, class_names, target_classes, line_thickness, bbox_opacity)
                st.image(res_b, use_column_width=True)
                st.write(f"객체 수: {len(labels_b)}")
        else:
            st.subheader(f"현재 이미지: {current_img_name}")
            res_a = draw_boxes(img.copy(), labels_a, class_names, target_classes, line_thickness, bbox_opacity)
            st.image(res_a, use_column_width=True)
            st.write(f"검출된 객체 수: {len(labels_a)}")

        # --- 통계 섹션 ---
        st.divider()
        st.subheader("📊 데이터셋 요약")
        # 간단한 통계 예시 (현재 이미지만)
        if labels_a:
            df_stats = pd.DataFrame([class_names[l['id']] for l in labels_a], columns=['Class'])
            st.bar_chart(df_stats['Class'].value_counts())
        else:
            st.info("현재 이미지에 라벨이 없습니다.")

else:
    st.warning("경로를 올바르게 입력해주세요.")

# --- 추가 기능 제안 ---
with st.expander("🛠️ 사용 팁"):
    st.write("""
    1. **비교 분석:** '라벨 폴더 B'를 입력하면 두 폴더의 라벨을 좌우로 비교할 수 있습니다.
    2. **오답 확인:** 특정 클래스만 필터링하여 오검출(FP)이나 미검출(FN)을 빠르게 찾으세요.
    3. **빠른 이동:** 키보드 화살표 키나 상단 슬라이더를 사용하여 이미지를 넘길 수 있습니다.
    """)
