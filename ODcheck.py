import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="문서 오버레이 비교기", layout="wide")

def reset_app():
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

st.title("📄 문서 이미지 겹쳐보기(오버레이) 정밀 비교")
st.markdown("하나의 이미지(좌: 초안, 우: 수정안)를 자동으로 반으로 나누어 겹쳐서 차이를 확인합니다.")

with st.sidebar:
    st.header("설정 및 도구")
    uploaded_file = st.file_uploader("좌우 병합된 문서 이미지 업로드", type=["jpg", "jpeg", "png"])
    
    st.divider()
    
    st.subheader("오버레이 설정")
    blend_mode = st.radio(
        "비교 모드 선택", 
        ["투명도 겹쳐보기 (Alpha Blend)", "차이점 강조 (Difference)"],
        help="'차이점 강조' 모드에서는 두 문서에서 픽셀이 다른 부분만 어둡게/색상으로 표시됩니다."
    )
    
    # 투명도 조절 슬라이더 (Alpha Blend 모드에서만 활성화)
    alpha = st.slider(
        "초안(왼쪽) 투명도 조절", 
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        disabled=(blend_mode != "투명도 겹쳐보기 (Alpha Blend)")
    )
    
    st.divider()
    if st.button("🔄 앱 초기화 (새로고침)", on_click=reset_app):
        pass

if uploaded_file:
    # 이미지를 OpenCV 배열로 변환
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 이미지를 반으로 분할
    h, w, _ = image.shape
    mid = w // 2
    left_img = image[:, :mid]
    right_img = image[:, mid:]
    
    # 좌우 이미지의 너비가 1px 정도 다를 수 있으므로 최소 너비에 맞춤
    min_w = min(left_img.shape[1], right_img.shape[1])
    left_img = left_img[:, :min_w]
    right_img = right_img[:, :min_w]

    st.subheader("🔍 겹쳐보기 결과")
    
    # 오버레이 처리 로직
    if blend_mode == "투명도 겹쳐보기 (Alpha Blend)":
        # left_img에 alpha 값 적용, right_img에 (1-alpha) 값 적용하여 합성
        blended = cv2.addWeighted(left_img, alpha, right_img, 1 - alpha, 0)
        st.image(blended, use_container_width=True, caption=f"초안 투명도: {alpha*100:.0f}% / 수정안 투명도: {(1-alpha)*100:.0f}%")
        
    elif blend_mode == "차이점 강조 (Difference)":
        # 두 이미지의 픽셀 값 차이의 절댓값을 구함 (동일한 부분은 검은색(0)이 됨)
        diff = cv2.absdiff(left_img, right_img)
        # 문서는 보통 흰 배경이므로, 보기 편하도록 색상을 반전시킴 (동일한 부분은 흰색, 다른 부분은 어둡게 표시)
        blended_diff = cv2.bitwise_not(diff)
        st.image(blended_diff, use_container_width=True, caption="변경된 글자나 줄 바꿈의 차이만 추출하여 표시합니다.")

    # 원본 이미지 참고용 Expander
    with st.expander("원본 분할 이미지 개별 확인하기"):
        col1, col2 = st.columns(2)
        with col1:
            st.image(left_img, caption="왼쪽 (초안)", use_container_width=True)
        with col2:
            st.image(right_img, caption="오른쪽 (수정안)", use_container_width=True)

else:
    st.info("👈 왼쪽 사이드바에서 비교할 이미지 파일을 업로드해 주세요.")
