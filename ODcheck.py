import streamlit as st
import cv2
import numpy as np
import pytesseract
import re

# 클라우드 배포 시 주석 처리 필요
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

st.set_page_config(page_title="스마트 문서 오버레이 비교기", layout="wide")

def reset_app():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

def preprocess_for_ocr(img):
    scale_factor = 2
    img_resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return thresh, scale_factor

def clean_text(text):
    text = str(text).replace(" ", "")
    clean = re.sub(r'[^가-힣a-zA-Z0-9]', '', text)
    return clean

def find_word_location(img, target_word):
    processed_img, scale = preprocess_for_ocr(img)
    custom_config = r'--oem 3 --psm 6'
    d = pytesseract.image_to_data(processed_img, lang='kor+eng', config=custom_config, output_type=pytesseract.Output.DICT)
    target_clean = clean_text(target_word)
    
    for i in range(len(d['text'])):
        current_text = clean_text(d['text'][i])
        if target_clean and target_clean in current_text:
            original_x = int(d['left'][i] / scale)
            original_y = int(d['top'][i] / scale)
            return original_x, original_y
    return None, None

def extract_diff_texts(diff_img, aligned_right_img):
    """차이점이 발생한 영역을 찾아 수정안(오른쪽)의 텍스트를 추출합니다."""
    # 1. 차이점을 흑백화 및 이진화 (다른 부분은 하얗게)
    gray_diff = cv2.cvtColor(diff_img, cv2.COLOR_BGR2GRAY)
    _, thresh_diff = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    
    # 2. 픽셀들을 문장 형태로 묶기 (가로로 길게 팽창)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 10))
    dilated = cv2.dilate(thresh_diff, kernel, iterations=2)
    
    # 3. 묶인 영역의 윤곽선(Contour) 찾기
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 위에서 아래로 읽도록 Y좌표 기준으로 정렬
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])
    
    extracted_texts = []
    # 수정안 이미지에 박스를 그릴 복사본
    boxed_img = aligned_right_img.copy()
    
    for (x, y, w, h) in bounding_boxes:
        # 노이즈(작은 점 등) 무시: 너비 20px, 높이 10px 이상만 취급
        if w > 20 and h > 10:
            # 인식률을 높이기 위해 상하좌우 여백(Padding)을 5px씩 주고 자름
            pad = 5
            y1, y2 = max(0, y - pad), min(aligned_right_img.shape[0], y + h + pad)
            x1, x2 = max(0, x - pad), min(aligned_right_img.shape[1], x + w + pad)
            
            crop = aligned_right_img[y1:y2, x1:x2]
            
            # 잘라낸 이미지에 OCR 전처리 적용
            crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            crop_resized = cv2.resize(crop_gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, crop_thresh = cv2.threshold(crop_resized, 150, 255, cv2.THRESH_BINARY)
            
            # psm 7: 해당 이미지를 '한 줄'의 텍스트로 강제 인식
            text = pytesseract.image_to_string(crop_thresh, lang='kor+eng', config='--psm 7').strip()
            
            # 의미 있는 문자가 포함된 경우만 결과에 추가
            if len(clean_text(text)) > 0:
                extracted_texts.append(text)
                # 수정안 이미지에 노란색 박스 그리기
                cv2.rectangle(boxed_img, (x, y), (x+w, y+h), (255, 215, 0), 2)
                
    return extracted_texts, boxed_img

st.title("🎯 스마트 문서 이미지 겹쳐보기 (자동 정렬 + 문장 추출)")
st.markdown("기준 단어로 이미지를 자동 정렬하고, 변경된 부분을 시각적으로 보여준 뒤 **수정된 문장을 직접 읽어냅니다.**")

with st.sidebar:
    st.header("설정 및 도구")
    uploaded_file = st.file_uploader("좌우 병합된 문서 이미지 업로드", type=["jpg", "jpeg", "png"])
    st.divider()
    st.subheader("정렬(Alignment) 설정")
    anchor_word = st.text_input("기준 단어 입력", value="제목")
    st.divider()
    st.subheader("오버레이 설정")
    blend_mode = st.radio("비교 모드 선택", ["차이점 강조 (Difference)", "투명도 겹쳐보기 (Alpha Blend)"])
    alpha = st.slider("초안 투명도 조절", 0.0, 1.0, 0.5, 0.05, disabled=(blend_mode != "투명도 겹쳐보기 (Alpha Blend)"))
    st.divider()
    if st.button("🔄 앱 초기화", on_click=reset_app):
        pass

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    h, w, _ = image.shape
    mid = w // 2
    left_img = image[:, :mid]
    right_img = image[:, mid:]
    
    min_w = min(left_img.shape[1], right_img.shape[1])
    left_img = left_img[:, :min_w]
    right_img = right_img[:, :min_w]

    st.subheader("🔍 분석 결과")
    
    with st.spinner(f"이미지 정렬 및 변경된 문장 추출 중..."):
        lx, ly = find_word_location(left_img, anchor_word)
        rx, ry = find_word_location(right_img, anchor_word)
        
        aligned_right_img = right_img.copy()
        
        if lx is not None and rx is not None:
            dx, dy = lx - rx, ly - ry
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            rows, cols = right_img.shape[:2]
            aligned_right_img = cv2.warpAffine(right_img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            st.success("✅ 문서 자동 정렬 완료")
        else:
            st.warning("⚠️ 기준 단어를 찾지 못해 원본 위치에서 비교합니다.")

        # 오버레이 처리 및 텍스트 추출을 위한 차이값 계산
        diff = cv2.absdiff(left_img, aligned_right_img)
        
        # 새로운 기능: 차이점 부분에서 텍스트 추출
        extracted_texts, boxed_img = extract_diff_texts(diff, aligned_right_img)

        # 화면 출력 분할 (좌측: 오버레이, 우측: 추출된 텍스트)
        col_img, col_text = st.columns([1.5, 1])
        
        with col_img:
            if blend_mode == "차이점 강조 (Difference)":
                blended_result = cv2.bitwise_not(diff)
                st.image(blended_result, use_container_width=True, caption="[시각적 차이점] 어둡게 표시된 부분이 변경된 영역입니다.")
            elif blend_mode == "투명도 겹쳐보기 (Alpha Blend)":
                blended_result = cv2.addWeighted(left_img, alpha, aligned_right_img, 1 - alpha, 0)
                st.image(blended_result, use_container_width=True, caption=f"[겹쳐보기] 초안 투명도: {alpha*100:.0f}%")
                
            st.image(boxed_img, use_container_width=True, caption="[영역 감지] 수정안 이미지에서 텍스트를 추출한 노란색 박스 영역입니다.")

        with col_text:
            st.markdown("### 📝 추출된 수정 문장")
            st.markdown("시각적으로 변화가 감지된 영역을 읽어낸 결과입니다.")
            
            if extracted_texts:
                for idx, text in enumerate(extracted_texts):
                    # 보기 편하도록 마크다운 인용구 스타일로 출력
                    st.info(f"**{idx + 1}번째 변경 영역:**\n> {text}")
            else:
                st.success("변경된 텍스트가 감지되지 않았습니다.")

else:
    st.info("👈 왼쪽 사이드바에서 비교할 이미지 파일을 업로드해 주세요.")
