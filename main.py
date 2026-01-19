import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

# 1. Load the Model
model = YOLO("best.pt")

# Open Video Source
cap = cv2.VideoCapture("fire-smoke.mp4")

# --- Chart Data Settings ---
history_length = 100 
fire_history = deque([0]*history_length, maxlen=history_length)
smoke_history = deque([0]*history_length, maxlen=history_length)

# Color Palette (BGR Format)
COLOR_FIRE = (0, 0, 255)       # Red
COLOR_SMOKE = (255, 255, 0)     # Cyan/Turquoise
COLOR_PANEL = (30, 30, 30)      # Dark Panel Background
COLOR_TEXT = (255, 255, 255)    # White

def draw_graph(panel, data, color, title, pos_y):
    """Draws a small line graph on the side panel."""
    cv2.putText(panel, title, (10, pos_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    cv2.rectangle(panel, (10, pos_y), (190, pos_y + 60), (60, 60, 60), 1)
    
    for i in range(1, len(data)):
        pt1_x = 10 + (i - 1) * 1.8 
        pt2_x = 10 + i * 1.8
        pt1_y = pos_y + 60 - int(data[i-1] * 60)
        pt2_y = pos_y + 60 - int(data[i] * 60)
        cv2.line(panel, (int(pt1_x), pt1_y), (int(pt2_x), pt2_y), color, 1)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # YOLOv8 Detection
    results = model(frame, conf=0.4, imgsz=320, verbose=False)
    
    current_max_fire = 0.0
    current_max_smoke = 0.0

    for r in results:
        for box in r.boxes:
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords
            
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id].lower()
            
            if "fire" in label:
                current_max_fire = max(current_max_fire, conf)
                color = COLOR_FIRE
            elif "smoke" in label:
                current_max_smoke = max(current_max_smoke, conf)
                color = COLOR_SMOKE
            else:
                color = (0, 255, 0)

            # Draw Bounding Boxes
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Label on TOP
            cv2.putText(frame, f"{label} %{int(conf*100)}", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # NEW: Intensity Text at the BOTTOM LEFT of the detection box
            if "smoke" in label:
                cv2.putText(frame, f"Intensity: %{int(conf*100)}", (x1, y2 + 20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Update Graph History
    fire_history.append(current_max_fire)
    smoke_history.append(current_max_smoke)

    # SIDE PANEL CREATION
    h, w, _ = frame.shape
    panel_w = 210
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
    panel[:] = COLOR_PANEL 

    # Panel Title
    cv2.putText(panel, "SYSTEM ANALYSIS", (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.5, COLOR_TEXT, 1)
    cv2.line(panel, (10, 45), (200, 45), (100, 100, 100), 1)

    # Draw Graphs
    draw_graph(panel, fire_history, COLOR_FIRE, "Fire Intensity", 80)
    draw_graph(panel, smoke_history, COLOR_SMOKE, "Smoke Intensity", 180)

    # Combine Images (Frame + Panel)
    final_ui = np.hstack((frame, panel))

    # Show Result
    cv2.imshow("Fire Tracking Dashboard ", final_ui)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()