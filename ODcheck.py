import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io

# Tesseract-OCR 경로 (윈도우 로컬 실행 시 필요하면 설정)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="AI 문서 비교 검수기", layout="wide")

def align_images_orb(img1, img2):
    """특징점 매칭을 이용한 이미지 자동 정렬"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * 0.15)]
    if len(good_matches) < 4: return img2
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]), borderValue=(255,255,255))

def process_analysis(image_rgb):
    """이미지 분석 전체 프로세스"""
    h, w, _ = image_rgb.shape
    mid = w // 2
    img_l = image_rgb[:, :mid]
    img_r = image_rgb[:, mid:]

    # 1. 자동 정렬
    aligned_r = align_images_orb(img_l, img_r)

    # 2. 차이점 계산
    diff = cv2.absdiff(img_l, aligned_r)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(cv2.GaussianBlur(diff_gray, (5, 5), 0), 30, 255, cv2.THRESH_BINARY)

    # 3. 텍스트 추출 및 박싱
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
    dilated = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])

    texts = []
    boxed_img = aligned_r.copy()
    for cnt in contours:
        x, y, wb, hb = cv2.boundingRect(cnt)
        if wb > 20 and hb > 10:
            roi = aligned_r[max(0, y-5):y+hb+5, max(0, x-5):x+wb+5]
            if roi.size == 0: continue
            text = pytesseract.image_to_string(roi, lang='kor+eng', config='--psm 7').strip()
            if text:
                texts.append(text)
                cv2.rectangle(boxed_img, (x, y), (x+wb, y+hb), (255, 0, 0), 2)

    return img_l, aligned_r, boxed_img, mask, texts

# --- 메인 화면 ---
st.title("🎯 AI 스마트 문서 비교기")

# 중요: Ctrl+V 사용법 안내
st.markdown("""
### 📋 이미지 입력 방법
1. 비교할 문서(좌우 병합본)를 캡처합니다 (`Win+Shift+S` 또는 `Cmd+Shift+4`).
2. **아래 'Browse files' 영역을 마우스로 한 번 클릭**합니다 (포커스 활성화).
3. **`Ctrl + V`** (맥은 `Cmd + V`)를 눌러 이미지를 바로 붙여넣으세요.
""")

uploaded_file = st.file_uploader("여기를 클릭한 후 Ctrl+V를 누르거나 파일을 드래그하세요", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 데이터 로드
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with st.spinner("이미지 분석 중..."):
        l_img, r_aligned, b_img, mask, results = process_analysis(image_rgb)

    # 결과 대시보드
    tab1, tab2 = st.tabs(["📊 결과 시각화", "📝 추출된 문장"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("원본(초안)")
            st.image(l_img, use_container_width=True)
        with c2:
            st.subheader("수정본(감지구역)")
            st.image(b_img, use_container_width=True)
        
        st.subheader("변동 영역 하이라이트")
        overlay = l_img.copy()
        overlay[mask > 0] = [255, 0, 0]
        st.image(overlay, use_container_width=True, caption="빨간색 부분이 수정된 위치입니다.")

    with tab2:
        st.subheader("🔍 감지된 변경 문장 리스트")
        if results:
            for i, text in enumerate(results):
                st.info(f"**구간 {i+1}:** {text}")
        else:
            st.success("변경된 텍스트가 없습니다.")
else:
    st.info("이미지를 기다리고 있습니다...")
