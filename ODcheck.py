import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ... (상단 설정 부분 동일)

def get_color(idx):
    """클래스 ID별로 고정된 색상을 반환 (Matplotlib 없이 구현)"""
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
        (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
        (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]
    return colors[idx % len(colors)]

def draw_boxes(image, labels, class_names, target_classes, thickness, opacity):
    overlay = image.copy()
    
    for label in labels:
        cls_id = label['id']
        if cls_id >= len(class_names) or class_names[cls_id] not in target_classes:
            continue
            
        color = get_color(cls_id) # 수정된 색상 함수 사용
        x1, y1, x2, y2 = label['bbox']
        
        # 박스 그리기
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        
        # 라벨 텍스트 배경 (가독성 향상)
        label_text = class_names[cls_id]
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
        cv2.putText(overlay, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0)

# ... (이후 로직 동일)
