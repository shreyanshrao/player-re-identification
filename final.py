from ultralytics import YOLO
import deep_sort_realtime

model = YOLO("best.pt") 
import cv2
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Initialize YOLOv8 model
detector = YOLO("yolov8s.pt")


tracker1 = DeepSort(max_age=30)
tracker2 = DeepSort(max_age=30)


embeddings_video1 = {}
embeddings_video2 = {}

def extract_embedding(image, bbox):
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))

    if x2 <= x1 or y2 <= y1:
        return None  

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None  

    crop = cv2.resize(crop, (64, 128))
    return crop.flatten() / 255.0


# Process one frame for detection and tracking
def process_frame(frame, tracker):
    detections = []
    results = detector(frame)[0]
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        if cls == 0:  # person class
            detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))
    tracks = tracker.update_tracks(detections, frame=frame)
    return tracks

# Process videos
def process_video(path, tracker, embedding_dict):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_skip = int(fps * 0.2)  # ~0.2 sec per frame

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            tracks = process_frame(frame, tracker)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = track.track_id
                bbox = track.to_ltrb()
                embedding = extract_embedding(frame, bbox)
                if embedding is not None:
                    embedding_dict[track_id] = embedding

        frame_count += 1
    cap.release()


# Run tracking and embedding
process_video("1st.mp4", tracker1, embeddings_video1)
process_video("2nd.mp4", tracker2, embeddings_video2)

# Match embeddings using cosine similarity
global_id_map = {}
global_id_counter = 0
used = set()

for id1, emb1 in embeddings_video1.items():
    best_score = 0
    best_id2 = None
    for id2, emb2 in embeddings_video2.items():
        if id2 in used:
            continue
        sim = cosine_similarity([emb1], [emb2])[0][0]
        if sim > best_score:
            best_score = sim
            best_id2 = id2
    if best_score > 0.8:  # similarity threshold
        global_id_map[id1] = global_id_counter
        global_id_map[best_id2] = global_id_counter
        used.add(best_id2)
        global_id_counter += 1

# Print the global ID mapping
print("Global ID Mapping Across Videos:")
for k, v in global_id_map.items():
    print(f"Local ID {k} → Global ID {v}")
