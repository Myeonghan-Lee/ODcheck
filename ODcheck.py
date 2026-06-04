import cv2
import numpy as np
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as ssim

def process_comparison(full_img, min_area, threshold_val):
    """이미지를 분석하여 차이점을 찾아내는 함수"""
    # 1. 이미지 분할
    h, w, _ = full_img.shape
    half_w = w // 2
    
    img_left = full_img[:, :half_w]
    img_right = full_img[:, half_w:half_w*2]

    # 크기 불일치 시 조정
    if img_left.shape != img_right.shape:
        img_right = cv2.resize(img_right, (img_left.shape[1], img_left.shape[0]))

    # 2. 그레이스케일 변환
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

    # 3. SSIM을 이용한 정밀 비교
    (score, diff) = ssim(gray_left, gray_right, full=True)
    diff = (diff * 255).astype("uint8")

    # 4. 임계값 및 노이즈 처리 (사용자 설정 가능하도록 개선)
    thresh = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel) # 노이즈 제거

    # 5. 차이점 영역 탐지
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result_img = img_right.copy()
    change_count = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area: # 설정한 면적 이상의 변화만 감지
            x, y, w_box, h_box = cv2.boundingRect(c)
            cv2.rectangle(result_img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 3)
            change_count += 1

    # 6. 히트맵 생성
    diff_color = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)

    return img_left, result_img, diff_color, score, change_count

# --- Streamlit UI 시작 ---
st.set_page_config(layout="wide", page_title="ODcheck - 이미지 비교 도구")
st.title("🔍 ODcheck: 수정 사항 자동 점검 도구")
st.write("이미지를 업로드하면 왼쪽(원본)과 오른쪽(수정본)을 비교하여 변경된 부분을 자동으로 찾아냅니다.")

# 사이드바 설정
st.sidebar.header("설정 (민감도 조절)")
min_area = st.sidebar.slider("감지할 최소 면적 (작을수록 예민)", 0, 500, 50)
threshold_val = st.sidebar.slider("임계값 (낮을수록 미세한 차이 감지)", 0, 200, 0)

uploaded_file = st.file_uploader("검토할 이미지 파일을 선택하세요 (좌우 분할 이미지)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # 이미지 로드
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # OpenCV BGR -> RGB 변환

    # 분석 실행
    with st.spinner('분석 중...'):
        left_part, right_part, diff_map, score, count = process_comparison(image, min_area, threshold_val)

    # 결과 요약
    st.success(f"분석 완료! 유사도: {score:.2%} | 감지된 수정 부분: {count}곳")

    # 결과 화면 출력
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. 원본 (왼쪽)")
        st.image(left_part, use_container_width=True)

    with col2:
        st.subheader("2. 수정 사항 확인")
        st.image(right_part, use_container_width=True)

    with col3:
        st.subheader("3. 차이점 히트맵")
        st.image(diff_map, use_container_width=True)
else:
    st.info("파일을 업로드하면 분석이 시작됩니다.")
