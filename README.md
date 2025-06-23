# 🎥 Multi-Camera Player Tracking and Re-Identification

This project tracks and identifies players across two different video feeds using computer vision techniques. It uses:

- **YOLOv8** for person detection
- **DeepSORT** for tracking each player in a video
- **Cosine Similarity** for matching players across different cameras
- **Global ID Mapping** to assign consistent IDs to the same player across feeds

---

## 📦 Requirements

Install all dependencies using:

pip install -r requirements.txt

I HAVE USED PRE DEFINED MODEL GIVEN BY YOU BEST.PT Which is fined tuned version of Ultralytics YOLO V11 trained for person and ball detection.
This project implements a complete end-to-end pipeline for tracking and identifying players across two different video feeds using computer vision. It combines YOLOv8 for real-time person detection, DeepSORT for within-video tracking, and cosine similarity on visual embeddings to match player identities across different camera angles.
if cosine simmilarity>0.8 we detect the player as same as the given player..
