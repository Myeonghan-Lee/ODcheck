import cv2
import numpy as np
import streamlit as st
from PIL import Image
from skimage.metrics import structural_similarity as ssim

# 페이지 설정
st.set_page_config(layout="wide", page_title="AI 이미지 검토 도구")

def analyze_differences(img_left, img_right, sensitivity):
    # 1. 그레이스케일 변환 및 노이즈 제거 (가우시안 블러)
    # 이미지 정렬 미세 오차로 인한 가짜 탐지를 줄이기 위함
    gray_left = cv2.cvtColor(img_left, cv2.COLOR_RGB2GRAY)
    gray_right = cv2.cvtColor(img_right, cv2.COLOR_RGB2GRAY)
    
    gray_left = cv2.GaussianBlur(gray_left, (5, 5), 0)
    gray_right = cv2.GaussianBlur(gray_right, (5, 5), 0)

    # 2. SSIM (구조적 유사성) 계산
    # score가 낮을수록 차이가 큰 지점
    (score, diff) = ssim(gray_left, gray_right, full=True)
    diff = (diff * 255).astype("uint8")

    # 3. 차이점 강조를 위한 임계값 처리
    # sensitivity 값이 낮을수록 더 예민하게 잡아냄
    thresh = cv2.threshold(diff, sensitivity, 255, cv2.THRESH_BINARY_INV)[1]
    
    # 4. 형태학적 변환 (미세한 점들을 뭉쳐서 하나의 영역으로 인식)
    kernel = np.ones((5, 5), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=2)
    
    # 5. 윤곽선 추출
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    output_img = img_right.copy()
    diff_boxes = []
    
    for i, c in enumerate(contours):
        area = cv2.contourArea(c)
        if area > 40:  # 너무 작은 노이즈는 무시
            x, y, w, h = cv2.boundingRect(c)
            # 수정본 이미지에 빨간 사각형 표시
            cv2.rectangle(output_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
            # 번호 표시 (안내 미흡 해결)
            cv2.putText(output_img, str(i+1), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            diff_boxes.append(f"수정사항 {i+1}: 좌표 ({x}, {y}) 주변 영역")

    return output_img, score, diff_boxes, thresh

# --- UI 부분 ---
st.title("🔍 자동 이미지 수정 사항 점검기")
st.markdown("""
이 도구는 **왼쪽(원본) | 오른쪽(수정본)**으로 구성된 이미지를 분석하여 변경된 부분을 자동으로 찾아냅니다.
""")

# 사이드바 설정
st.sidebar.header("⚙️ 분석 설정")
sensitivity = st.sidebar.slider("탐지 민감도 (낮을수록 미세한 차이 탐지)", 0, 200, 150)
st.sidebar.info("정확하게 찾지 못한다면 민감도를 조절해 보세요.")

uploaded_file = st.file_uploader("이미지 업로드 (PNG, JPG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # 이미지 처리
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    h, w, _ = img_array.shape
    half_w = w // 2
    
    left_part = img_array[:, :half_w]   # 원본
    right_part = img_array[:, half_w:]  # 수정본 (또는 결과물)

    # 분석 실행
    result_img, similarity, log, mask = analyze_differences(left_part, right_part, sensitivity)

    # 결과 요약 레이아웃
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("✅ 수정 사항 시각화")
        st.image(result_img, caption=f"분석 결과 (유사도: {similarity:.2%})", use_container_width=True)
        
    with col2:
        st.subheader("📝 점검 리스트")
        if log:
            st.write(f"총 **{len(log)}군데**의 수정 사항이 발견되었습니다.")
            for item in log:
                st.write(f"- {item}")
        else:
            st.success("원본과 완벽히 일치하거나 수정 사항이 없습니다!")

    # 상세 분석 (히트맵)
    with st.expander("차이점 추출 마스크(Mask) 보기"):
        st.image(mask, caption="차이점이 집중된 구역 (흰색 부분)", use_container_width=True)

else:
    st.warning("분석할 이미지를 업로드해 주세요.")
