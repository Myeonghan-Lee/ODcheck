import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
from streamlit_paste_button import paste_image_button
import io

# Tesseract 경로 설정 (윈도우 사용자만 필요시 주석 해제)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="AI 문서 비교 검수기", layout="wide")

def align_images_orb(img1, img2):
    """두 이미지의 특징점을 찾아 자동으로 수평/수직 정렬"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)

    good_matches = matches[:int(len(matches) * 0.15)]
    if len(good_matches) < 4:
        return img2

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    aligned_img = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]), 
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return aligned_img

def process_analysis(image_rgb):
    """이미지 반분, 정렬, 차이점 탐지, OCR 추출"""
    h, w, _ = image_rgb.shape
    mid = w // 2
    img_l = image_rgb[:, :mid]
    img_r = image_rgb[:, mid:]

    # 1. 자동 정렬
    aligned_r = align_images_orb(img_l, img_r)

    # 2. 차이점 마스크 생성
    diff = cv2.absdiff(img_l, aligned_r)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
    _, mask = cv2.threshold(diff_blur, 30, 255, cv2.THRESH_BINARY)

    # 3. 텍스트 영역 검출 및 OCR
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
    dilated = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1]) # 상단부터 정렬

    results = []
    boxed_img = aligned_r.copy()

    for cnt in contours:
        x, y, wb, hb = cv2.boundingRect(cnt)
        if wb > 20 and hb > 10:
            roi = aligned_r[max(0, y-5):y+hb+5, max(0, x-5):x+wb+5]
            if roi.size == 0: continue
            
            # OCR (한글+영어)
            text = pytesseract.image_to_string(roi, lang='kor+eng', config='--psm 7').strip()
            if text:
                results.append(text)
                cv2.rectangle(boxed_img, (x, y), (x+wb, y+hb), (255, 0, 0), 2)

    return img_l, aligned_r, boxed_img, mask, results

# --- UI 화면 구성 ---
st.title("📑 AI 스마트 문서 비교 검수기")
st.info("비교할 문서(좌우 병합본)를 캡처한 후, 아래 '클립보드 이미지 붙여넣기' 버튼을 클릭하세요.")

# 사이드바 레이아웃
with st.sidebar:
    st.header("입력 도구")
    # 방법 1: 붙여넣기 버튼 (추천)
    pasted_output = paste_image_button(
        label="📋 클립보드 이미지 붙여넣기",
        background_color="#FF4B4B",
        hover_color="#D33636",
        errors="ignore"
    )
    st.divider()
    # 방법 2: 파일 업로드 (백업)
    uploaded_file = st.file_uploader("또는 이미지 파일 업로드", type=["png", "jpg", "jpeg"])

# 데이터 로드 로직
input_img = None

if pasted_output.image_data is not None:
    # 붙여넣은 이미지 처리
    input_img = np.array(pasted_output.image_data.convert("RGB"))
elif uploaded_file is not None:
    # 업로드한 이미지 처리
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    input_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# 분석 및 출력
if input_img is not None:
    with st.spinner("이미지 정렬 및 문장 분석 중..."):
        l_img, r_aligned, b_img, diff_mask, texts = process_analysis(input_img)

    tab1, tab2 = st.tabs(["📊 시각적 분석", "📝 추출된 변경 문장"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("초안 (왼쪽)")
            st.image(l_img, use_container_width=True)
        with c2:
            st.subheader("수정안 (검지된 영역)")
            st.image(b_img, use_container_width=True)
        
        st.subheader("픽셀 차이 하이라이트 (빨간색)")
        overlay = l_img.copy()
        overlay[diff_mask > 0] = [255, 0, 0]
        st.image(overlay, use_container_width=True)

    with tab2:
        st.subheader("🔍 감지된 텍스트 리스트")
        if texts:
            for i, t in enumerate(texts):
                st.info(f"**변동 구간 {i+1}:** {t}")
        else:
            st.success("변경 사항이 없습니다.")
else:
    st.warning("👈 사이드바에서 이미지를 붙여넣거나 업로드해 주세요.")
