import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io

# 1. Tesseract 설정 (클라우드 배포 시에는 환경 설정 파일에 추가)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="AI 문서 통합 검수기", layout="wide")

def align_images_orb(img1, img2):
    """ORB 특징점 매칭을 이용한 정밀 자동 정렬"""
    # 그레이스케일 변환
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # ORB 탐지기 생성
    orb = cv2.ORB_create(3000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # 특징점 매칭
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 상위 매칭점 추출
    good_matches = matches[:int(len(matches) * 0.15)]
    if len(good_matches) < 4:
        return img2 # 매칭 실패 시 원본 반환

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 호모그래피 행렬 계산 및 이미지 변환
    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    aligned_img = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]), 
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return aligned_img

def get_analysis(image_rgb):
    """이미지 분할, 정렬, 차이점 추출 및 OCR 통합 처리"""
    h, w, _ = image_rgb.shape
    mid = w // 2
    left_img = image_rgb[:, :mid]
    right_img = image_rgb[:, mid:]

    # 자동 정렬 실행
    aligned_right = align_images_orb(left_img, right_img)

    # 차이점 계산 (노이즈 제거 포함)
    diff = cv2.absdiff(left_img, aligned_right)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)
    _, mask = cv2.threshold(diff_blur, 30, 255, cv2.THRESH_BINARY)

    # 변경 영역 박스 및 OCR 추출
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 10))
    dilated = cv2.dilate(mask, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 위에서 아래로 정렬
    contours = sorted(contours, key=lambda c: cv2.boundingRect(c)[1])
    
    extracted_texts = []
    boxed_img = aligned_right.copy()

    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        if w_box > 15 and h_box > 10:
            # 해당 영역 잘라내기 (약간의 여백)
            roi = aligned_right[max(0, y-5):y+h_box+5, max(0, x-5):x+w_box+5]
            if roi.size == 0: continue
            
            # OCR 인식 (한글 + 영어)
            text = pytesseract.image_to_string(roi, lang='kor+eng', config='--psm 7').strip()
            if text:
                extracted_texts.append(text)
                cv2.rectangle(boxed_img, (x, y), (x+w_box, y+h_box), (255, 0, 0), 2)

    return left_img, aligned_right, boxed_img, mask, extracted_texts

# --- 메인 화면 UI ---
st.title("🎯 AI 스마트 문서 비교기 (캡처 지원)")
st.markdown("""
**사용 방법:**
1. 비교할 두 문서가 좌우로 배치된 이미지를 준비합니다.
2. 아래 업로드 박스를 클릭한 후, **`Ctrl + V`**를 눌러 스크린샷을 바로 붙여넣으세요.
3. 또는 이미지 파일을 드래그하여 업로드하세요.
""")

# 사이드바 설정
with st.sidebar:
    st.header("⚙️ 설정")
    st.info("스크린샷 캡처 팁: 'Win+Shift+S'로 영역 선택 후 이 앱에 'Ctrl+V' 하세요.")
    if st.button("🔄 새로고침"):
        st.rerun()

# 이미지 입력부 (파일 업로드 + 클립보드 붙여넣기 지원)
uploaded_file = st.file_uploader("이미지를 업로드하거나 붙여넣으세요 (좌우 병합본)", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # 이미지 로드
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with st.spinner("이미지를 자동 정렬하고 문장을 분석 중입니다..."):
        left_img, aligned_right, boxed_img, mask, texts = get_analysis(image_rgb)

    # 탭 구성
    tab1, tab2 = st.tabs(["📊 비교 시각화", "📝 변경 내용 추출"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("초안 (왼쪽)")
            st.image(left_img, use_container_width=True)
        with col2:
            st.subheader("수정안 (검지된 영역)")
            st.image(boxed_img, use_container_width=True)
        
        st.divider()
        st.subheader("변동 영역 하이라이트")
        # 원본 이미지 위에 빨간색으로 차이점 오버레이
        overlay = left_img.copy()
        overlay[mask > 0] = [255, 0, 0] # 빨간색 강조
        st.image(overlay, use_container_width=True, caption="빨간색으로 표시된 부분이 변경된 텍스트 위치입니다.")

    with tab2:
        st.subheader("🔍 변동 문장 리스트")
        if texts:
            for i, txt in enumerate(texts):
                st.info(f"**구간 {i+1}:** {txt}")
            
            # 텍스트 복사용 편의 기능
            all_text = "\n".join(texts)
            st.download_button("텍스트 파일로 저장", all_text, file_name="diff_result.txt")
        else:
            st.success("감지된 텍스트 변경 사항이 없습니다.")
else:
    st.info("👈 비교할 이미지를 업로드하거나 클립보드에서 붙여넣으세요.")
