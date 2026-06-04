import streamlit as st
import cv2
import numpy as np
import pytesseract
import re
from PIL import Image

# Tesseract 경로 설정 (필요시)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="AI 문서 비교 검수기", layout="wide")

def align_images_orb(img1, img2):
    """ORB 특징점 매칭을 이용한 자동 이미지 정렬"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 상위 10% 매칭점만 사용
    good_matches = matches[:int(len(matches) * 0.1)]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    aligned_img = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    
    return aligned_img

def get_diff_mask(img1, img2, threshold=30):
    """차이점 마스크 생성 (노이즈 제거 포함)"""
    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    # 미세한 차이는 무시하기 위해 블러 처리
    diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
    _, diff_mask = cv2.threshold(diff_blur, threshold, 255, cv2.THRESH_BINARY)
    return diff_mask

def extract_text_from_diff(diff_mask, target_img):
    """차이가 있는 영역에서만 텍스트 추출"""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
    dilated = cv2.dilate(diff_mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    boxed_img = target_img.copy()
    
    # Y 좌표 순으로 정렬 (위에서 아래로)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 10:
            roi = target_img[max(0, y-5):y+h+5, max(0, x-5):x+w+5]
            if roi.size == 0: continue
            
            # OCR 인식률 향상을 위한 전처리
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            text = pytesseract.image_to_string(roi_gray, lang='kor+eng', config='--psm 7').strip()
            
            if text:
                results.append(text)
                cv2.rectangle(boxed_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
    return results, boxed_img

# --- UI Layout ---
st.title("📑 스마트 문서 변동사항 탐지기")
st.markdown("두 이미지의 픽셀 단위 차이를 분석하여 **수정된 문장만 골라냅니다.**")

uploaded_file = st.sidebar.file_uploader("비교할 이미지 업로드 (좌우 병합본)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 이미지 로드
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    full_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    full_img = cv2.cvtColor(full_img, cv2.COLOR_BGR2RGB)

    # 반으로 나누기
    h, w, _ = full_img.shape
    mid = w // 2
    img_left = full_img[:, :mid]
    img_right = full_img[:, mid:]

    # 처리 시작
    with st.spinner("이미지 정렬 및 분석 중..."):
        # 1. 자동 정렬
        try:
            aligned_right = align_images_orb(img_left, img_right)
            st.sidebar.success("✅ 자동 정렬 성공")
        except:
            aligned_right = img_right
            st.sidebar.warning("⚠️ 자동 정렬 실패 (기본 위치 사용)")

        # 2. 차이점 추출
        diff_mask = get_diff_mask(img_left, aligned_right)
        texts, boxed_img = extract_text_from_diff(diff_mask, aligned_right)

    # 결과 표시
    tab1, tab2 = st.tabs(["📊 분석 결과 시각화", "📝 추출된 텍스트"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("초안 (왼쪽)")
            st.image(img_left, use_container_width=True)
        with c2:
            st.subheader("수정안 (오른쪽 - 감지 구역)")
            st.image(boxed_img, use_container_width=True)
        
        st.subheader("변동 영역 하이라이트")
        # 차이점을 빨간색으로 강조
        overlay = img_left.copy()
        overlay[diff_mask > 0] = [255, 0, 0] # 차이점을 빨간색으로
        st.image(overlay, use_container_width=True, caption="빨간색으로 표시된 부분이 변경된 위치입니다.")

    with tab2:
        st.subheader("🔍 변경 내용 리스트")
        if texts:
            for i, t in enumerate(texts):
                st.info(f"**변경 구간 {i+1}:** {t}")
        else:
            st.write("감지된 변경 사항이 없습니다.")
else:
    st.info("사이드바에서 이미지를 업로드하세요.")
