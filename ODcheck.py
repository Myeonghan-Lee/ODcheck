import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import io

# --- 1. 페이지 설정 ---
st.set_page_config(layout="wide", page_title="ODcheck Pro - Multi Source")

if 'history' not in st.session_state:
    st.session_state.history = []

# --- 2. 유틸리티 함수 ---
def parse_yolo_labels(label_text, img_w, img_h):
    """YOLO 텍스트 데이터를 리스트로 변환"""
    labels = []
    if not label_text:
        return labels
    lines = label_text.strip().split('\n')
    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) == 5:
            cls, x, y, w, h = map(float, parts)
            x1 = int((x - w/2) * img_w)
            y1 = int((y - h/2) * img_h)
            x2 = int((x + w/2) * img_w)
            y2 = int((y + h/2) * img_h)
            labels.append({'id': i, 'class': int(cls), 'bbox': [x1, y1, x2, y2], 'raw': [cls, x, y, w, h]})
    return labels

def draw_bboxes(image, labels, selected_id=None):
    """이미지에 바운딩 박스 시각화 (선택된 박스는 강조)"""
    img_draw = image.copy()
    for label in labels:
        x1, y1, x2, y2 = label['bbox']
        is_selected = (label['id'] == selected_id)
        color = (255, 0, 0) if is_selected else (0, 255, 0)
        thickness = 4 if is_selected else 2
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, thickness)
        # 텍스트 배경
        label_str = f"ID:{label['id']} Cls:{label['class']}"
        cv2.putText(img_draw, label_str, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return img_draw

# --- 3. 사이드바: 입력 소스 선택 ---
st.sidebar.title("📥 입력 소스 설정")
input_mode = st.sidebar.radio("이미지 불러오기 방식", ["파일 업로드", "카메라 캡처", "로컬 경로(기존)"])

img_array = None
label_input = ""

if input_mode == "파일 업로드":
    uploaded_file = st.sidebar.file_uploader("이미지 파일 선택", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

elif input_mode == "카메라 캡처":
    camera_file = st.sidebar.camera_input("이미지 캡처")
    if camera_file:
        file_bytes = np.asarray(bytearray(camera_file.read()), dtype=np.uint8)
        img_array = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

else: # 로컬 경로
    img_dir = st.sidebar.text_input("이미지 폴더 경로", "data/images")
    if os.path.exists(img_dir):
        files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        sel_file = st.sidebar.selectbox("파일 선택", files)
        if sel_file:
            img_array = cv2.imread(os.path.join(img_dir, sel_file))
            img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)

# 라벨 입력 섹션
st.sidebar.markdown("---")
st.sidebar.subheader("🏷️ 라벨링 데이터 (YOLO)")
label_method = st.sidebar.radio("라벨 불러오기 방식", ["텍스트 직접 붙여넣기", "파일 업로드"])

if label_method == "텍스트 직접 붙여넣기":
    label_input = st.sidebar.text_area("YOLO 라벨 내용 입력", placeholder="0 0.5 0.5 0.2 0.2", height=150)
else:
    uploaded_lbl = st.sidebar.file_uploader("라벨 파일(.txt) 선택", type=['txt'])
    if uploaded_lbl:
        label_input = uploaded_lbl.getvalue().decode("utf-8")

# --- 4. 메인 화면 구성 ---
if img_array is not None:
    h, w, _ = img_array.shape
    labels = parse_yolo_labels(label_input, w, h)
    
    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.subheader("🖼️ 원본 및 탐지 결과")
        # 객체 선택용 리스트 (찾기 편하도록 ID 부여)
        selected_id = st.selectbox("집중 점검할 객체 선택 (ID)", options=[l['id'] for l in labels] if labels else [None])
        
        canvas = draw_bboxes(img_array, labels, selected_id)
        st.image(canvas, use_column_width=True)

    with col2:
        st.subheader("🔍 정밀 분석 및 수정")
        
        if labels and selected_id is not None:
            target = next(l for l in labels if l['id'] == selected_id)
            x1, y1, x2, y2 = target['bbox']
            
            # 1. 자동 확대 기능 (Zoom-in)
            pad = 50
            crop = img_array[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
            st.image(crop, caption=f"ID {selected_id} 확대 화면", use_column_width=True)
            
            # 2. 데이터 요약 정보 테이블
            df = pd.DataFrame(labels)
            st.dataframe(df[['id', 'class', 'raw']], use_container_width=True)
            
            # 3. 수정 및 기록
            new_cls = st.number_input("클래스 ID 수정", value=target['class'], step=1)
            if st.button("수정 사항 기록"):
                change_log = f"ID {selected_id}: {target['class']} -> {new_cls}"
                st.session_state.history.append(change_log)
                st.success(f"기록됨: {change_log}")

        # 4. 수정 이력 보기
        st.markdown("---")
        with st.expander("📝 최근 수정 이력", expanded=True):
            if st.session_state.history:
                for h in reversed(st.session_state.history[-10:]):
                    st.write(f"• {h}")
            else:
                st.info("수정 이력이 없습니다.")

else:
    st.info("이미지를 업로드하거나 카메라를 사용해 보세요.")
