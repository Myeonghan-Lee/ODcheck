import streamlit as st
import cv2
import numpy as np
import pytesseract
import base64
from io import BytesIO
from PIL import Image
import streamlit.components.v1 as components

# Tesseract 경로 설정 (필요시)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="AI 문서 비교 검수기", layout="wide")

# --- JavaScript Screen Capture Component ---
def screen_capture_js():
    """브라우저의 getDisplayMedia API를 사용하여 화면을 캡처하는 JS 코드"""
    js_code = """
    <div style="margin-bottom: 10px;">
        <button id="captureBtn" style="
            background-color: #ff4b4b; color: white; border: none; 
            padding: 10px 20px; border-radius: 5px; cursor: pointer; font-weight: bold;">
            📸 화면 캡처(공유) 시작
        </button>
        <video id="video" style="display:none;" autoplay></video>
        <canvas id="canvas" style="display:none;"></canvas>
    </div>

    <script>
    const captureBtn = document.getElementById('captureBtn');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');

    captureBtn.addEventListener('click', async () => {
        try {
            // 화면 공유 요청
            const stream = await navigator.mediaDevices.getDisplayMedia({ video: true });
            video.srcObject = stream;
            
            // 스트림이 준비될 때까지 잠시 대기
            await new Promise(r => setTimeout(r, 500));
            
            // 캔버스에 그리기
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);
            
            // Base64 데이터 추출
            const imageData = canvas.toDataURL('image/png');
            
            // Streamlit으로 데이터 전송
            window.parent.postMessage({
                type: 'streamlit:setComponentValue',
                value: imageData
            }, '*');

            // 스트림 종료
            stream.getTracks().forEach(track => track.stop());
        } catch (err) {
            console.error("Error: " + err);
        }
    });
    </script>
    """
    return components.html(js_code, height=60)

# --- 기존 로직 함수들 ---
def align_images_orb(img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(2000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    good_matches = matches[:int(len(matches) * 0.1)]
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]), borderValue=(255,255,255))

def process_analysis(image_rgb):
    """이미지를 분석하여 결과 반환"""
    h, w, _ = image_rgb.shape
    mid = w // 2
    img_left = image_rgb[:, :mid]
    img_right = image_rgb[:, mid:]

    # 정렬
    try:
        aligned_right = align_images_orb(img_left, img_right)
    except:
        aligned_right = img_right

    # 차이점 마스크
    diff = cv2.absdiff(img_left, aligned_right)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    _, diff_mask = cv2.threshold(cv2.GaussianBlur(diff_gray, (5, 5), 0), 30, 255, cv2.THRESH_BINARY)

    # 텍스트 추출
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
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
            text = pytesseract.image_to_string(cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY), lang='kor+eng', config='--psm 7').strip()
            if text:
                texts.append(text)
                cv2.rectangle(boxed_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
    return img_left, aligned_right, boxed_img, diff_mask, texts

# --- 메인 화면 구성 ---
st.title("📑 AI 문서 비교 & 실시간 캡처 검수기")
st.markdown("이미지를 업로드하거나, **화면을 직접 캡처**하여 문서의 변경 사항을 즉시 확인하세요.")

# 입력 방식 선택 (사이드바)
input_mode = st.sidebar.radio("입력 방식 선택", ["이미지 파일 업로드", "스크린 캡처 사용"])

input_image = None

if input_mode == "이미지 파일 업로드":
    uploaded_file = st.sidebar.file_uploader("좌우 병합 이미지 업로드", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

else:
    st.sidebar.info("1. 아래 버튼 클릭\n2. 캡처할 창 선택\n3. '공유' 클릭")
    captured_data = screen_capture_js() # JS 컴포넌트 실행
    
    if captured_data:
        # Base64 데이터를 넘파이 이미지로 변환
        header, encoded = captured_data.split(",", 1)
        data = base64.b64decode(encoded)
        image = Image.open(BytesIO(data))
        input_image = np.array(image.convert("RGB"))

# --- 분석 및 결과 출력 ---
if input_image is not None:
    left, right_aligned, boxed, mask, texts = process_analysis(input_image)

    tab1, tab2 = st.tabs(["📊 시각적 분석", "📝 변경된 문장 추출"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("초안 (Original)")
            st.image(left, use_container_width=True)
        with c2:
            st.subheader("수정안 (Detected)")
            st.image(boxed, use_container_width=True)
        
        st.subheader("변동 영역 하이라이트")
        overlay = left.copy()
        overlay[mask > 0] = [255, 0, 0]
        st.image(overlay, use_container_width=True, caption="빨간색 표시: 변경된 픽셀 위치")

    with tab2:
        st.subheader("🔍 추출된 텍스트 리스트")
        if texts:
            for i, t in enumerate(texts):
                st.info(f"**구간 {i+1}:** {t}")
        else:
            st.success("변경 사항이 감지되지 않았습니다.")
else:
    st.warning("👈 이미지를 업로드하거나 '화면 캡처 시작' 버튼을 눌러주세요.")
