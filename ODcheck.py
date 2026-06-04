import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

# --- 페이지 설정 ---
st.set_page_config(page_title="ODcheck Pro - Advanced Toolkit", layout="wide")

st.title("🔍 ODcheck Pro: Object Detection Quality Control")
st.markdown("""
이 도구는 데이터셋의 품질을 시각적으로 점검하고 변경 사항을 추적하기 위해 설계되었습니다.
""")

# --- 사이드바: 설정 및 경로 ---
with st.sidebar:
    st.header("📂 데이터 경로 설정")
    img_dir = st.text_input("이미지 폴더 경로", value="./data/images")
    label_dir = st.text_input("라벨(.txt) 폴더 경로", value="./data/labels")
    
    st.header("⚙️ 시각화 설정")
    box_thickness = st.slider("박스 두께", 1, 10, 2)
    show_label_text = st.checkbox("클래스 이름 표시", value=True)
    
    st.header("🎯 필터링")
    filter_option = st.selectbox("이미지 필터", ["전체 보기", "라벨 있음", "라벨 없음(Empty)"])

# --- 데이터 로드 함수 ---
def get_file_list(path, ext=(".jpg", ".png", ".jpeg")):
    if not os.path.exists(path):
        return []
    return sorted([f for f in os.listdir(path) if f.lower().endswith(ext)])

def read_yolo(label_path, img_w, img_h):
    bboxes = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = list(map(float, line.strip().split()))
                if len(parts) == 5:
                    cls, x, y, w, h = parts
                    # 복원: 중심좌표 -> 좌상단/우하단
                    x1 = int((x - w/2) * img_w)
                    y1 = int((y - h/2) * img_h)
                    x2 = int((x + w/2) * img_w)
                    y2 = int((y + h/2) * img_h)
                    bboxes.append({'cls': int(cls), 'bbox': (x1, y1, x2, y2)})
    return bboxes

def draw_boxes(image, bboxes, thickness, show_text):
    img_draw = image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for obj in bboxes:
        c = colors[obj['cls'] % len(colors)]
        x1, y1, x2, y2 = obj['bbox']
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), c, thickness)
        if show_text:
            cv2.putText(img_draw, f"ID: {obj['cls']}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
    return img_draw

# --- 메인 로직 ---
img_files = get_file_list(img_dir)

if not img_files:
    st.warning("이미지 폴더를 확인해주세요. 파일을 찾을 수 없습니다.")
else:
    # 세션 상태로 현재 인덱스 관리
    if 'idx' not in st.session_state:
        st.session_state.idx = 0

    # 네비게이션 버튼
    col1, col2, col3, col4 = st.columns([1, 1, 4, 2])
    with col1:
        if st.button("⬅️ 이전"):
            st.session_state.idx = max(0, st.session_state.idx - 1)
    with col2:
        if st.button("다음 ➡️"):
            st.session_state.idx = min(len(img_files) - 1, st.session_state.idx + 1)
    with col4:
        st.write(f"**진행도:** {st.session_state.idx + 1} / {len(img_files)}")

    # 현재 파일 처리
    target_img_name = img_files[st.session_state.idx]
    target_img_path = os.path.join(img_dir, target_img_name)
    target_label_path = os.path.join(label_dir, target_img_name.rsplit('.', 1)[0] + ".txt")

    # 이미지 로드
    img = cv2.imread(target_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape

    # 라벨 로드
    bboxes = read_yolo(target_label_path, w, h)

    # UI 레이아웃: 이미지 표시 및 정보
    tab1, tab2 = st.tabs(["🖼️ 시각적 검수", "📊 데이터셋 통계"])

    with tab1:
        c1, c2 = st.columns(2)
        
        # 원본 이미지 (수정 전 가정)
        with c1:
            st.subheader("원본 데이터 (Original)")
            canvas_raw = draw_boxes(img, bboxes, box_thickness, show_label_text)
            st.image(canvas_raw, use_column_width=True)
            
        # 비교 이미지 또는 정보 (추후 수정 기능 확장 가능)
        with c2:
            st.subheader("상세 정보 (Details)")
            st.info(f"파일명: {target_img_name}")
            st.write(f"해상도: {w}x{h}")
            if bboxes:
                df_bboxes = pd.DataFrame(bboxes)
                st.write(f"검출된 객체 수: {len(bboxes)}")
                st.dataframe(df_bboxes)
            else:
                st.warning("이 이미지는 라벨 정보가 없습니다.")

    with tab2:
        st.subheader("전체 클래스 분포")
        # 간단한 통계 계산 (샘플링)
        all_labels = []
        for f in os.listdir(label_dir)[:100]: # 성능상 100개만 우선 분석
            if f.endswith(".txt"):
                with open(os.path.join(label_dir, f), 'r') as lf:
                    for line in lf:
                        all_labels.append(int(line.split()[0]))
        
        if all_labels:
            df_counts = pd.Series(all_labels).value_counts().sort_index()
            st.bar_chart(df_counts)
        else:
            st.write("통계 데이터를 불러올 수 없습니다.")

# --- 변경 사항 추적용 메모 섹션 ---
st.divider()
st.subheader("📝 검수 메모 및 변경 로그")
log_text = st.text_area("현재 이미지에 대한 특이사항을 기록하세요.", "")
if st.button("로그 저장"):
    with open("audit_log.txt", "a") as f:
        f.write(f"[{target_img_name}] {log_text}\n")
    st.success("로그가 기록되었습니다.")
