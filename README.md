# 📑 AI 문서 오버레이 비교기 (AI Document Diff Tool)

이 프로젝트는 두 장의 문서 이미지(초안 및 수정안)를 분석하여 **위치 정렬, 시각적 차이점 강조, 그리고 변경된 부분의 텍스트를 자동으로 추출**하는 Streamlit 기반 웹 애플리케이션입니다.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=flat-square&logo=opencv&logoColor=white)

---

## ✨ 주요 기능

1.  **자동 이미지 정렬 (Automatic Alignment):** 
    *   ORB(Oriented FAST and Rotated BRIEF) 특징점 매칭을 통해 스캔이나 캡처 시 발생한 미세한 위치 차이를 자동으로 보정합니다.
2.  **시각적 차이점 분석 (Visual Diff):**
    *   두 문서를 겹쳐서 변경된 부분을 빨간색 하이라이트 또는 투명도 조절(Alpha Blend)을 통해 직관적으로 보여줍니다.
3.  **수정 문장 자동 추출 (OCR Extraction):**
    *   Tesseract OCR을 활용하여 시각적 변화가 감지된 영역 내의 텍스트(한글/영어)를 인식하고 리스트 형태로 제공합니다.
4.  **사용자 친화적 인터페이스:**
    *   Streamlit을 통한 간편한 파일 업로드 및 실시간 파라미터 조절 기능을 제공합니다.

---

## 🛠 설치 방법

### 1. 필수 시스템 도구 설치 (Tesseract-OCR)

본 프로젝트는 OCR 기능을 위해 `Tesseract-OCR` 엔진이 시스템에 설치되어 있어야 합니다.

*   **Windows:** [Tesseract installer](https://github.com/UB-Mannheim/tesseract/wiki)에서 `.exe` 파일을 다운로드하여 설치합니다. (설치 시 'Additional script data'에서 'Korean' 필히 체크)
*   **Linux (Ubuntu):**
    ```bash
    sudo apt update
    sudo apt install tesseract-ocr tesseract-ocr-kor
    ```

### 2. 파이썬 환경 설정

```bash
# 저장소 복제
git clone https://github.com/your-username/document-comparison-tool.git
cd document-comparison-tool

# 필수 라이브러리 설치
pip install -r requirements.txt
```

**requirements.txt 내용:**
```text
streamlit
opencv-python-headless
numpy
Pillow
pytesseract
```

---

## 🚀 사용 방법

1.  **애플리케이션 실행:**
    ```bash
    streamlit run app.py
    ```
2.  **이미지 업로드:**
    *   왼쪽 사이드바에서 초안과 수정안이 좌우로 합쳐진 이미지를 업로드합니다.
3.  **결과 확인:**
    *   **분석 결과 시각화 탭:** 정렬된 문서와 변경 위치 하이라이트를 확인합니다.
    *   **추출된 텍스트 탭:** 변경된 영역에서 인식된 실제 문구 리스트를 확인합니다.

---

## 📂 프로젝트 구조

*   `app.py`: 메인 Streamlit 애플리케이션 코드
*   `requirements.txt`: 필요한 파이썬 라이브러리 목록
*   `README.md`: 프로젝트 설명 문서

---

## ⚙️ 주요 알고리즘 설명

*   **Image Registration:** `cv2.ORB_create`와 `cv2.findHomography`를 사용하여 문서의 회전 및 밀림 현상을 수학적으로 계산하여 일치시킵니다.
*   **Diff Detection:** `cv2.absdiff`로 두 이미지의 차이를 구한 후, 노이즈 제거(Gaussian Blur)와 이진화(Thresholding)를 거쳐 실제 텍스트 변경 구간을 탐색합니다.
*   **Contour Analysis:** 변경된 픽셀들을 팽창(Dilation)시켜 단어/문장 단위의 바운딩 박스를 생성하고 해당 구역만 OCR을 수행하여 정확도를 높였습니다.

---

## ⚠️ 유의 사항

*   **OCR 경로:** Windows 사용자의 경우 `app.py` 상단에서 `pytesseract.pytesseract.tesseract_cmd` 경로를 본인의 설치 경로에 맞게 수정해야 할 수 있습니다.
*   **이미지 품질:** 해상도가 너무 낮거나 노이즈가 심한 스캔본의 경우 OCR 인식률이 떨어질 수 있습니다.

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자유롭게 수정 및 배포가 가능합니다.
