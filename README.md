# AI Projects with MediaPipe and OpenCV

A collection of real-time computer vision and AI applications built with **MediaPipe**, **OpenCV**, **PyTorch** and **Flask**. Each project has its own folder with source code, a `README.md` (features, setup and usage) and, where available, a demo video.

## Projects

| # | Project | Description | Key Tech |
|---|---|---|---|
| 1 | [OCR and Text-to-Speech Application](1_OCR_and_Text_to_Speech_Application) | Extracts text from images and PDFs and reads it aloud, for accessibility and document processing | Tesseract OCR, gTTS, Flask |
| 2 | [PoseTrack: Real-Time Body and Hand Motion Capture](2_PoseTrack_Real_Time_Body_and_Hand_Motion_Capture) | Detects body and hand landmarks from a webcam, plots them in 2D and records/replays motion data | MediaPipe, OpenCV |
| 4 | [Background Removal API](4_Background_Removal_API) | Web app that removes an uploaded image's background with semantic segmentation | DeepLabV3, PyTorch, Flask |
| 5 | [Virtual Whiteboard](5_Virtual_Whiteboard) | Real-time drawing canvas controlled entirely by hand gestures | MediaPipe Hands, OpenCV |
| 6 | [Object Detection with Wikipedia Integration](6_reega) | Real-time object detection that fetches and speaks a Wikipedia summary of each detected object | YOLOv5, PyTorch, Wikipedia API, TTS |
| 7 | [Real-Time Facial Expression Detection](7_Real_Time_Facial_Expression_Detection_using_Mediapipe_and_OpenCV) | Classifies facial expressions from geometric relationships between face-mesh landmarks | MediaPipe FaceMesh, OpenCV |
| 8 | [Driver Drowsiness and Distraction Detection](8_Driver_Drowsiness_and_Distraction_Detection_System) | Tracks eye aspect ratio (EAR) for drowsiness and detects phone use for distraction, with alerts | MediaPipe, OpenCV, YOLOv8 |
| 9 | [DeepLabV3 Segmentation with Text Addition](9_DeepLabV3_Image_Segmentation_with_Foreground_Background_Separation_and_Text_Addition) | Separates foreground and background and places custom text behind the subject | DeepLabV3 (torchvision), Flask |
| 10 | [Hand Gesture Filter Control](10_Hand_Gesture_Filter_Control_with_Real_Time_Visual_Effects) | Switches between real-time visual filters on a webcam feed with pinch gestures | MediaPipe Hands, OpenCV, Tkinter |

## Getting Started

```bash
git clone https://github.com/Manimohan05/AI_projects_MediaPipe_and_OpenCV.git
cd AI_projects_MediaPipe_and_OpenCV/<project-folder>
pip install -r requirements.txt   # where provided
```

Then follow the instructions in that project's `README.md`. Most projects need a webcam.

## Tools
Python · MediaPipe · OpenCV · PyTorch / torchvision · Ultralytics YOLO · Flask · Tesseract · gTTS
