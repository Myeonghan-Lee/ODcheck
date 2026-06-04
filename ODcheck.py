import streamlit as st
import cv2
import numpy as np
import pytesseract
from streamlit_paste_button import paste_image_button  # 클립보드 지원 라이브러리

# Tesseract 경로 설정 (필요 시)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="AI 문서 비교 검수기", layout="wide")

def align_images_orb(img1, img2):
    """ORB 특징점 매칭을 이용한 자동 이미지 정렬"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    good_matches = matches[:int(len(matches) * 0.1)]
    if len(good_matches) < 4: return img2 # 매칭점 부족 시 원본 반환
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    aligned_img = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]), 
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))
    return aligned_img

def process_and_compare(image):
    """이미지를 반으로 나누고 비교 분석 수행"""
    # 1. 이미지 분할 (좌/우)
    h, w, _ = image.shape
    mid = w // 2
    img_left = image[:, :mid]
    img_right = image[:, mid:]

    # 2. 자동 정렬 및 차이점 추출
    aligned_right = align_images_orb(img_left, img_right)
    diff = cv2.absdiff(img_left, aligned_right)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, diff_mask = cv2.threshold(cv2.GaussianBlur(diff_gray, (5, 5), 0), 30, 255, cv2.THRESH_BINARY)

    # 3. 텍스트 추출 (박싱)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    dilated = cv2.dilate(diff_mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    results = []
    boxed_img = aligned_right.copy()
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1]) # 상단부터 정렬

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 25 and h > 15:
            roi = aligned_right[max(0, y-5):y+h+5, max(0, x-5):x+w+5]
            text = pytesseract.image_to_string(roi, lang='kor+eng', config='--psm 7').strip()
            if text:
                results.append(text)
                cv2.rectangle(boxed_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    return img_left, boxed_img, results, diff_mask

# --- UI 레이아웃 ---
st.title("🚀 고속 문서 비교 검수기")
st.markdown("윈도우 캡처(`Win+Shift+S`) 후 아래 버튼을 눌러 바로 비교하세요!")

# 입력 방식 선택
input_method = st.sidebar.radio("입력 방식 선택", ["클립보드 붙여넣기", "파일 업로드"])
target_image = None

if input_method == "클립보드 붙여넣기":
    st.info("이미지를 캡처한 후 아래 'Paste Image' 버튼을 클릭하고 Ctrl+V 하세요.")
    paste_result = paste_image_button("📋 클립보드 이미지 붙여넣기")
    if paste_result.image_data is not None:
        # PIL Image를 OpenCV 형식으로 변환
        target_image = np.array(paste_result.image_data.convert("RGB"))
        
else:
    uploaded_file = st.sidebar.file_uploader("좌우 병합된 이미지 업로드", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        target_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)

# 분석 및 결과 표시
if target_image is not None:
    left_img, boxed_img, texts, mask = process_and_compare(target_image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("원본(초안)")
        st.image(left_img, use_container_width=True)
    with col2:
        st.subheader("수정본(감지됨)")
        st.image(boxed_img, use_container_width=True)

    st.divider()
    
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("🔴 변경 위치 하이라이트")
        overlay = left_img.copy()
        overlay[mask > 0] = [255, 0, 0]
        st.image(overlay, use_container_width=True)
    
    with c2:
        st.subheader("📝 추출된 수정 문구")
        if texts:
            for i, txt in enumerate(texts):
                st.info(f"**[{i+1}]** {txt}")
        else:
            st.success("변경 사항이 감지되지 않았습니다.")
else:
    st.warning("분석할 이미지를 가져와 주세요.")
