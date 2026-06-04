import cv2
import numpy as np
import pyautogui
from PIL import Image, ImageGrab
import tkinter as tk
from tkinter import filedialog
from skimage.metrics import structural_similarity as ssim

class VisualDiffChecker:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw() # 메인 창은 숨김

    def load_image(self):
        """파일 탐색기를 통해 이미지를 업로드합니다."""
        file_path = filedialog.askopenfilename(title="이미지 선택", 
                                               filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path:
            return None
        return cv2.imread(file_path)

    def capture_screen(self):
        """현재 화면을 캡처합니다."""
        print("3초 후 화면을 캡처합니다...")
        self.root.after(3000)
        screenshot = pyautogui.screenshot()
        return cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)

    def process_comparison(self, full_img):
        """이미지를 반으로 나누고 차이점을 분석합니다."""
        h, w, _ = full_img.shape
        half_w = w // 2
        
        # 1. 이미지 분할 (왼쪽: 원본, 오른쪽: 수정본)
        img_left = full_img[:, :half_w]
        img_right = full_img[:, half_w:half_w*2]

        # 크기가 다를 경우 조정
        img_right = cv2.resize(img_right, (img_left.shape[1], img_left.shape[0]))

        # 2. 그레이스케일 변환
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)

        # 3. SSIM을 이용한 차이점 계산 (정확도 핵심)
        (score, diff) = ssim(gray_left, gray_right, full=True)
        diff = (diff * 255).astype("uint8")

        # 4. 임계값 처리 및 노이즈 제거
        thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel) # 미세 구멍 채우기

        # 5. 윤곽선(Contour) 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 결과 이미지 복사
        result_img = img_right.copy()
        change_count = 0

        for c in contours:
            area = cv2.contourArea(c)
            if area > 50:  # 너무 작은 노이즈는 무시
                x, y, w_box, h_box = cv2.boundingRect(c)
                cv2.rectangle(result_img, (x, y), (x + w_box, y + h_box), (0, 0, 255), 2)
                change_count += 1

        # 6. 결과 시각화 구성
        # 원본 | 결과(박스표시) | 차이맵(히트맵 형태)
        diff_color = cv2.applyColorMap(thresh, cv2.COLORMAP_JET)
        combined = np.hstack((img_left, result_img, diff_color))

        # 안내 문구 추가
        status_text = f"Detected Changes: {change_count} areas. Similarity: {score:.2%}"
        cv2.putText(combined, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return combined

    def run(self):
        print("1: 이미지 업로드, 2: 화면 캡처")
        choice = input("선택하세요: ")
        
        img = None
        if choice == '1':
            img = self.load_image()
        elif choice == '2':
            img = self.capture_screen()
        
        if img is not None:
            result = self.process_comparison(img)
            cv2.imshow("Comparison Result (Left: Original | Mid: Result | Right: Diff Map)", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("이미지를 불러오지 못했습니다.")

if __name__ == "__main__":
    checker = VisualDiffChecker()
    checker.run()
