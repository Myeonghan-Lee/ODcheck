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
    """OCR 인식률을 극대화하기 위한 이미지 전처리"""
    # 1. 해상도 2배 확대 (글자를 크게 만들어 인식률 향상)
    scale_factor = 2
    img_resized = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)
    
    # 2. 흑백 변환 및 이진화 (배경은 하얗게, 글자는 검게)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    return thresh, scale_factor

def clean_text(text):
    """특수기호와 공백을 모두 제거하고 순수 문자만 반환"""
    text = str(text).replace(" ", "")
    # 한글, 영문, 숫자만 남기고 모두 제거 (예: '[제목]:' -> '제목')
    clean = re.sub(r'[^가-힣a-zA-Z0-9]', '', text)
    return clean

def find_word_location(img, target_word):
    """향상된 OCR로 이미지에서 특정 단어의 x, y 좌표를 찾습니다."""
    processed_img, scale = preprocess_for_ocr(img)
    
    # --psm 6: 하나의 텍스트 블록으로 인식하도록 강제 (공문서 형태에 적합)
    custom_config = r'--oem 3 --psm 6'
    d = pytesseract.image_to_data(processed_img, lang='kor+eng', config=custom_config, output_type=pytesseract.Output.DICT)
    
    target_clean = clean_text(target_word)
    
    for i in range(len(d['text'])):
        current_text = clean_text(d['text'][i])
        
        # 정제된 텍스트 안에 목표 단어가 포함되어 있다면
        if target_clean and target_clean in current_text:
            # 해상도를 2배 키웠으므로, 원본 이미지에 맞는 좌표로 다시 나누어 반환
            original_x = int(d['left'][i] / scale)
            original_y = int(d['top'][i] / scale)
            return original_x, original_y
            
    return None, None

st.title("🎯 스마트 문서 이미지 겹쳐보기 (자동 정렬)")
st.markdown("특정 기준 단어(예: '제목')의 위치를 찾아 두 문서의 미세한 위치 차이를 자동으로 교정하여 겹쳐 보여줍니다.")

with st.sidebar:
    st.header("설정 및 도구")
    uploaded_file = st.file_uploader("좌우 병합된 문서 이미지 업로드", type=["jpg", "jpeg", "png"])
    
    st.divider()
    
    st.subheader("정렬(Alignment) 설정")
    anchor_word = st.text_input("기준 단어 입력", value="제목", help="이 단어의 위치를 기준으로 좌우 이미지를 정렬합니다.")
    
    st.divider()
    
    st.subheader("오버레이 설정")
    blend_mode = st.radio(
        "비교 모드 선택", 
        ["차이점 강조 (Difference)", "투명도 겹쳐보기 (Alpha Blend)"]
    )
    
    alpha = st.slider(
        "초안(왼쪽) 투명도 조절", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        disabled=(blend_mode != "투명도 겹쳐보기 (Alpha Blend)")
    )
    
    st.divider()
    if st.button("🔄 앱 초기화", on_click=reset_app):
        pass

if uploaded_file:
    # 이미지 로드 및 분할
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

    st.subheader("🔍 분석 및 겹쳐보기 결과")
    
    with st.spinner(f"'{anchor_word}' 단어를 찾아 이미지를 정렬하는 중입니다... (OCR 분석 중)"):
        # 1. 기준 단어 좌표 찾기
        lx, ly = find_word_location(left_img, anchor_word)
        rx, ry = find_word_location(right_img, anchor_word)
        
        aligned_right_img = right_img.copy()
        
        # 2. 좌표를 성공적으로 찾았다면 오차 계산 및 이미지 이동
        if lx is not None and rx is not None:
            dx = lx - rx  # 가로 이동량
            dy = ly - ry  # 세로 이동량
            
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            rows, cols = right_img.shape[:2]
            aligned_right_img = cv2.warpAffine(right_img, M, (cols, rows), borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
            
            st.success(f"✅ '{anchor_word}' 단어를 기준으로 정렬 완료! (보정: 가로 {dx}px, 세로 {dy}px 이동)")
        else:
            # 실패 원인을 더 구체적으로 안내
            error_msg = f"⚠️ 양쪽 문서에서 '{anchor_word}' 단어를 모두 찾지는 못했습니다. "
            if lx is None and rx is None:
                error_msg += "양쪽 모두 실패했습니다."
            elif lx is None:
                error_msg += "왼쪽(초안)에서 찾지 못했습니다."
            else:
                error_msg += "오른쪽(수정안)에서 찾지 못했습니다."
            error_msg += " ('관련'이나 '붙임' 등 다른 단어를 시도해 보세요.)"
            st.warning(error_msg)

        # 3. 오버레이 처리
        if blend_mode == "차이점 강조 (Difference)":
            diff = cv2.absdiff(left_img, aligned_right_img)
            blended_result = cv2.bitwise_not(diff)
            st.image(blended_result, use_container_width=True, caption="변경된 부분만 어둡게/색상으로 나타납니다.")
            
        elif blend_mode == "투명도 겹쳐보기 (Alpha Blend)":
            blended_result = cv2.addWeighted(left_img, alpha, aligned_right_img, 1 - alpha, 0)
            st.image(blended_result, use_container_width=True, caption=f"초안 투명도: {alpha*100:.0f}%")

    with st.expander("원본 및 정렬된 이미지 개별 확인하기"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(left_img, caption="왼쪽 원본 (초안)", use_container_width=True)
        with col2:
            st.image(right_img, caption="오른쪽 원본 (수정안)", use_container_width=True)
        with col3:
            st.image(aligned_right_img, caption="위치 보정된 수정안", use_container_width=True)

else:
    st.info("👈 왼쪽 사이드바에서 비교할 이미지 파일을 업로드해 주세요.")
