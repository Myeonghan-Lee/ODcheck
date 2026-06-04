import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io

# Tesseract 경로 설정 (윈도우 환경에서 필요시 주석 해제)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="AI 문서 비교 검수기", layout="wide")

def align_images_orb(img1, img2):
    """ORB 특징점 매칭을 이용한 자동 이미지 정렬"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    
    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None: return img2
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    good_matches = matches[:int(len(matches) * 0.1)]
    if len(good_matches) < 4: return img2
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    if M is None: return img2
    
    aligned_img = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]), 
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return aligned_img

def process_analysis(image):
    """이미지 분할, 정렬, 차이점 및 텍스트 추출 통합 처리"""
    # 1. 이미지 분할
    h, w, _ = image.shape
    mid = w // 2
    img_left = image[:, :mid]
    img_right = image[:, mid:]

    # 2. 자동 정렬
    aligned_right = align_images_orb(img_left, img_right)

    # 3. 차이점 마스크 생성
    diff = cv2.absdiff(img_left, aligned_right)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
    _, diff_mask = cv2.threshold(diff_blur, 30, 255, cv2.THRESH_BINARY)

    # 4. 텍스트 추출용 영역 검출
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    dilated = cv2.dilate(diff_mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    texts = []
    boxed_img = aligned_right.copy()
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 20 and h > 10:
            roi = aligned_right[max(0, y-5):y+h+5, max(0, x-5):x+w+5]
            if roi.size == 0: continue
            # OCR (한국어+영어)
            text = pytesseract.image_to_string(roi, lang='kor+eng', config='--psm 7').strip()
            if text:
                texts.append(text)
                cv2.rectangle(boxed_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
    return img_left, aligned_right, boxed_img, texts, diff_mask

# --- UI 부분 ---
st.title("📑 AI 스마트 문서 비교기")
st.markdown("""
**사용 방법:** 
1. `Win + Shift + S` (윈도우) 또는 `Cmd + Shift + 4` (맥)로 비교할 문서(좌우 병합본)를 캡처합니다.
2. 아래의 **파일 업로드 박스를 클릭**한 후, **`Ctrl + V`**를 눌러 이미지를 붙여넣으세요!
""")

# 별도 라이브러리 없이 기본 uploader 사용
uploaded_file = st.file_uploader("이미지 파일을 선택하거나 클립보드에서 붙여넣기 하세요.", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 이미지 로드 및 변환
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    with st.spinner("이미지를 분석하고 있습니다..."):
        left_img, aligned_right, boxed_img, texts, mask = process_analysis(img_array)

    # 결과 레이아웃
    tab1, tab2 = st.tabs(["🔍 시각적 비교", "📝 변경된 문장 리스트"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("원본 (초안)")
            st.image(left_img, use_container_width=True)
        with col2:
            st.subheader("수정본 (감지된 영역)")
            st.image(boxed_img, use_container_width=True)
            
        st.subheader("🔴 변경 지점 하이라이트")
        overlay = left_img.copy()
        # 차이점 마스크가 있는 부분을 빨간색으로 강조
        overlay[mask > 0] = [255, 0, 0]
        st.image(overlay, use_container_width=True, caption="초안 위에 수정된 부분을 빨간색으로 표시했습니다.")

    with tab2:
        st.subheader("추출된 수정 내용")
        if texts:
            for i, t in enumerate(texts):
                st.info(f"**[{i+1}]** {t}")
        else:
            st.success("감지된 텍스트 변경 사항이 없습니다.")
else:
    st.info("💡 캡처 후 이 창에서 Ctrl+V를 누르면 즉시 분석이 시작됩니다.")
