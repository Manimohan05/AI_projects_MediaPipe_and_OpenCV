# Projects Overview

This folder contains multiple projects. Below is an overview of each project:

## .git


## 10_Hand_Gesture_Filter_Control_with_Real_Time_Visual_Effects

---

# Hand Gesture Filter Control with Real-Time Visual Effects & Image Filter Application

This repository contains two interactive applications that apply real-time visual effects using **OpenCV** and **Mediapipe** for hand gesture recognition. Users can control the application either by using hand gestures (for the Hand Gesture Filter Control) or by selecting filters from a dropdown menu (for the Image Filter Application). Both projects provide a dynamic user experience where filters are applied in real-time on webcam video feeds.

## 1. Hand Gesture Filter Control with Real-Time Visual Effects

This application uses **Mediapipe** to track hand gestures, enabling control over real-time video filters with gestures like pinches or finger touches. **OpenCV** is used for applying various visual effects to the webcam feed, and users can cycle through different filters using simple hand movements.

### Features:
- **Hand Gesture Recognition:** Detects gestures like pinches and finger touches to trigger real-time filter changes.
- **Multiple Filters:** Includes a collection of filters (e.g., Grayscale, Edge Detection, Colormap Effects, Blur, and more).
- **Gesture-Controlled Filter Cycling:** Filter changes occur on detecting a pinch gesture, with a delay to avoid rapid switching.
- **Interactive Feedback:** The active filter is displayed on the live video feed.
- **Real-Time Video Processing:** Webcam input is processed in real-time using **OpenCV**.

### Supported Filters:
- Grayscale
- Canny Edge Detection
- Gaussian Blur
- Colormap Effects (Jet, Ocean, Pink, Winter, Cool)
- Bilateral Filter
- Median Blur
- Laplacian Edge Detection
- Sobel Edge Detection
- Noise Reduction (Optional)
- And more...

### Use Cases:
- **Interactive Art Creation:** Create dynamic art with hand gestures.
- **Live Streaming:** Apply filters to a live video stream for an engaging viewer experience.
- **Gesture-Controlled Interfaces:** Use hand gestures to control applications or games.
- **Fitness or Wellness Apps:** Track movements during workouts with customized filters.

---

## 2. Image Filter Application with Real-Time Webcam Feed

This application provides a GUI for applying various image filters to a live webcam feed. It uses **OpenCV** to process video frames and **Tkinter** to display the webcam feed and filter options. Users can select filters from a dropdown and control the webcam feed using buttons for real-time effects.

### Features:
- **Webcam Feed with Filters:** Real-time video processing using **OpenCV** to apply filters to a webcam feed.
- **Multiple Filters:** A variety of filters, including edge detection, sepia tone, emboss, noise reduction, and thresholding.
- **Filter Selection:** Users can choose a filter from a dropdown menu and view its description.
- **Filter Navigation:** Users can cycle through filters using **Next** and **Back** buttons.
- **Interactive GUI:** Built with **Tkinter**, featuring buttons to start/stop the webcam and display filter information.

### Supported Filters:
- Laplacian Edge Detection
- Sobel Edge Detection
- Scharr Filter
- Black-and-White Silhouette
- Noise Reduction
- Sepia Tone
- Emboss Effect
- Adaptive Threshold
- Otsu Threshold
- And more...

### Use Cases:
- **Webcam Effects:** Apply filters to a live webcam feed for entertainment, streaming, or tutorials.
- **Image Processing Education:** Demonstrates various computer vision techniques like edge detection and noise reduction.
- **Creative Art Effects:** Apply artistic effects for a unique visual experience.

---

## Requirements:
For both applications, you need the following dependencies:

- **Python 3.x**
- **OpenCV**
- **Mediapipe** (for Hand Gesture Filter Control)
- **Tkinter** (pre-installed with Python)
- **Pillow (PIL)**
- **NumPy**

You can install the required dependencies using the following command:

```bash
pip install opencv-python mediapipe numpy pillow
```

---

## Installation Instructions:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/hand-gesture-filters.git
   ```

2. **Install dependencies:**

   Install all dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**

   - For **Hand Gesture Filter Control**, run the following command:
   
     ```bash
     python main.py
     ```

   - For **Image Filter Application**, run this command:
   
     ```bash
     python filter.py
     ```

---

## Contributions:
Feel free to contribute to this project! You can:
- Create pull requests to add more filters or features.
- Report bugs or issues.
- Suggest new features to improve the application.

---

## Acknowledgments:
- **OpenCV**: For providing powerful computer vision tools.
- **Mediapipe**: For efficient hand gesture recognition.
- **Tkinter**: For building the graphical user interface.

---

This repository aims to demonstrate the power of computer vision and interactive user interfaces in creating engaging and dynamic applications. Enjoy experimenting with the filters and hand gestures!

---

This **README** provides an overview of both applications, installation instructions, and features for the users to easily navigate and use the respective projects.

### Videos:
- [filter control.mp4](./10_Hand_Gesture_Filter_Control_with_Real_Time_Visual_Effects\filter control.mp4)
- [hand_gesture_filter_control_implementation.mp4](./10_Hand_Gesture_Filter_Control_with_Real_Time_Visual_Effects\hand_gesture_filter_control_implementation.mp4)


## 1_OCR_and_Text_to_Speech_Application
# OCR-and-Text-to-Speech-Application
This project demonstrates the integration of Optical Character Recognition (OCR) and Text-to-Speech (TTS) technologies into a simple yet impactful Python application. The application extracts text from images and converts it into speech, offering a practical solution for accessibility needs and document processing.


## 2_PoseTrack_Real_Time_Body_and_Hand_Motion_Capture
# PoseTrack-Real-Time-Body-and-Hand-Motion-Capture
This project leverages MediaPipe and OpenCV to detect body and hand landmarks in real-time from a webcam feed. The application visualizes detected landmarks on both a 2D plot and the video feed, with the added functionality of recording and replaying landmark data.

Features
Real-Time Detection: Detects body and hand landmarks using MediaPipe's Pose and Hands solutions.
Landmark Visualization: Visualizes landmarks on a 2D plot and overlays them on the video feed.
Recording Mode: Records landmark data for body and hands to a JSON file.
Replay Mode: Replays previously recorded landmark movements.
User-Friendly Controls:
W: Start recording.
E: Stop recording and save data to a JSON file.
A: Replay recorded data.
Q: Quit the application.


Requirements
Python 3.7+
OpenCV
MediaPipe
Matplotlib
NumPy

Use the key controls to interact with the application:
Press W to start recording landmarks.
Press E to stop recording and save to a JSON file.
Press A to replay a recording.
Press Q to quit.

File Structure
body_hand_landmarks.py: Main script for detection, recording, and replay.
Recorded Data: Saved as recording_YYYYMMDD_HHMMSS.json.
Future Improvements
Add support for multiple camera inputs.
Include 3D visualization for landmarks.
Improve replay controls (e.g., pause, rewind, fast-forward).
Export recorded data to other formats (e.g., CSV).


Acknowledgments
MediaPipe for the robust pose and hand detection solutions.
OpenCV for video processing.
Matplotlib for 2D plotting.

### Videos:
- [Real_Time_Body_and_Hand_Motion_Capture.mkv](./2_PoseTrack_Real_Time_Body_and_Hand_Motion_Capture\Real_Time_Body_and_Hand_Motion_Capture.mkv)


## 4_Background_Removal_API
# Background Removal API

This is a Flask-based web application that allows users to upload an image, remove its background using the **DeepLabV3 model**, and download the processed image with a white background. The application leverages deep learning models to perform semantic segmentation, specifically the **DeepLabV3 model** from **PyTorch**.

## Features
- **Image Upload**: Upload any image to the server.
- **Background Removal**: The DeepLabV3 model removes the background and keeps the foreground.
- **Processed Image**: View the processed image with a white background.
- **Download**: Option to download the processed image in **PNG** format.

## Technologies Used
- **Flask**: A lightweight Python web framework for creating the web application.
- **PyTorch**: A deep learning library for model inference (DeepLabV3 used for semantic segmentation).
- **OpenCV**: Used for image processing.
- **NumPy**: For array manipulations.
- **PIL (Pillow)**: For image manipulation and saving the final output.
- **Werkzeug**: Utility for secure file handling.

## Installation

### Prerequisites:
- **Python 3.6+**
- **pip** (Python package installer)

### Steps to set up the project:
1. Clone this repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/background-removal-api.git
   cd background-removal-api

### Directory Structure:

```
background-removal-api/
│
├── app.py                # Main Flask application
├── static/
│   └── uploads/          # Directory to store uploaded and processed images
├── templates/
│   ├── index.html        # HTML template for the homepage
│
└── requirements.txt      # Python dependencies
```


## 5_Virtual_Whiteboard
# Gesture-Controlled Drawing Canvas 🎨  
**A real-time drawing application controlled entirely by hand gestures, powered by AI and computer vision.**

---

## 🌟 Overview  
This project allows users to interact with a digital canvas using their hand gestures, eliminating the need for traditional input devices like a mouse or stylus.  

Key gestures include:  
- **Index Finger** 🖊️: Acts as the drawing tool.  
- **Fist Gesture** ✊: Pauses the drawing action.  
- **Open Palm Gesture** 🖐️: Clears the canvas.

The application is built using **Python**, leveraging **MediaPipe** for hand tracking and **OpenCV** for creating the drawing interface.

---

## 🎯 Features  
- **Gesture-Based Control**: Intuitive gestures for drawing, pausing, and clearing the canvas.  
- **Real-Time Performance**: Fast and accurate hand tracking using MediaPipe.  
- **Interactive Feedback**: Visual indications of gestures and drawing actions on the canvas.  
- **Lightweight and Scalable**: Simple yet powerful implementation, adaptable to other applications like gaming or education.  

---

## 🛠️ Technologies Used  
- **Python**: Core language for implementing the project logic.  
- **MediaPipe**: For efficient hand detection and tracking.  
- **OpenCV**: For creating and managing the drawing canvas.  
- **NumPy**: To handle matrix operations for processing hand landmarks.  

---

## 🚀 Installation & Setup  

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-github-profile/gesture-controlled-drawing.git
   cd gesture-controlled-drawing

.
## 🖥️ How It Works
**Hand Detection**:
The application uses MediaPipe to detect and track hand landmarks in real-time.

**Gesture Recognition:**

Index finger movement is identified by checking specific hand landmark positions.
A closed fist and open palm are recognized based on landmark arrangements.

**Canvas Interaction:**

Detected gestures are translated into actions on the OpenCV canvas.
The drawing is updated dynamically based on the finger's coordinates.

## 🤔 Future Enhancements
**Multi-Finger Gestures:** Enable complex gestures for advanced interactions.
**Shape Recognition:** Automatically detect and draw geometric shapes based on gestures.
**AR/VR Integration:** Bring the canvas into a 3D environment for immersive experiences.


## 🤝 Contribution
**Contributions are welcome! Feel free to open an issue or submit a pull request.**


## 6_reega
# Object Detection with YOLOv5 and Wikipedia Integration

This Python script enables real-time object detection using YOLOv5 and provides additional information about the detected objects by fetching data from Wikipedia. The script utilizes computer vision libraries such as OpenCV and PyTorch, along with the Wikipedia API and text-to-speech capabilities.

## Installation

Before running the script, ensure you have the required dependencies installed. You can install them using pip:

```bash
pip install opencv-python torch torchvision pillow wikipedia-api pyttsx3
```

## Usage

1. Clone the repository or download the script from [GitHub]().

2. Run the script by executing the following command:

```bash
python object_detection.py
```

3. The script will start capturing video from the default camera and perform real-time object detection.

4. Detected objects will be displayed on the screen with their class names.

5. Press the following keys for additional functionalities:

   - Press `1` to display detailed information about the detected objects.
   - Press `2` to read only the class names of the detected objects.
   - Press `3` to resume real-time object detection if paused.
   - Press `4` to read the whole information about the detected objects.

6. To exit the script, press the `q` key.

## Acknowledgments

- YOLOv5: This script utilizes the YOLOv5 model for object detection. For more information about YOLOv5, refer to the [YOLOv5 GitHub repository](https://github.com/ultralytics/yolov5).

- Wikipedia API: The Wikipedia API is used to fetch information about detected objects from Wikipedia. For more information about the Wikipedia API, refer to the [Wikipedia-API documentation](https://pypi.org/project/Wikipedia-API/).

- Text-to-speech: The script uses the `pyttsx3` library for text-to-speech functionality. For more information about `pyttsx3`, refer to the [pyttsx3 documentation](https://pyttsx3.readthedocs.io/en/latest/).

## Notes

- Adjust confidence thresholds and other parameters as needed for optimal object detection performance.
- Ensure a stable internet connection for Wikipedia information retrieval.
- This script is intended for educational and experimental purposes and may require modifications for specific use cases or environments.


## 7_Real_Time_Facial_Expression_Detection_using_Mediapipe_and_OpenCV
### **GitHub Repository Description**

---

# **Real-Time Facial Expression Detection using Mediapipe and OpenCV**

This project demonstrates a real-time facial expression recognition system using **Mediapipe FaceMesh** and **OpenCV**. It detects and categorizes facial expressions based on geometric relationships between facial landmarks and visualizes the results with a user-friendly interface.

---

## **Features**

- **Facial Landmark Detection**:  
  Identifies and tracks up to 468 facial landmarks in real time using Mediapipe's FaceMesh.

- **Expression Recognition**:  
  Classifies expressions into categories like:
  - **Neutral**
  - **Smiling**
  - **Surprised**
  - **Angry**

- **Visual Feedback**:  
  Displays live webcam feed alongside a plotted visualization of detected facial landmarks.

- **Real-Time Analysis**:  
  Processes video frames at runtime for immediate feedback on detected expressions.

---

## **Use Cases**

1. **Human-Computer Interaction**:  
   Emotion-aware systems for personalized user experiences.

2. **Retail and Marketing**:  
   Analyze customer reactions to products or advertisements.

3. **Healthcare and Therapy**:  
   Non-intrusive emotional state monitoring.

4. **Education**:  
   Evaluate student engagement during e-learning sessions.

---

## **Technologies Used**

- **OpenCV**:  
  For video capture and image display.

- **Mediapipe**:  
  Provides FaceMesh for facial landmark detection.

- **NumPy**:  
  Used for geometric calculations between landmark points.

---

## **How It Works**

1. **Capture Video**:  
   Uses OpenCV to stream live video from a webcam.

2. **Detect Landmarks**:  
   Mediapipe FaceMesh identifies facial landmarks in real time.

3. **Analyze Expressions**:  
   Calculates distances between key landmarks (e.g., eyes, lips) to determine expressions.

4. **Visualize Results**:  
   Displays the webcam feed and a blank canvas highlighting the detected landmarks.

---

## **Installation**

1. Clone this repository:  
   ```bash
   git clone https://github.com/<your-username>/real-time-facial-expression-detection.git
   cd real-time-facial-expression-detection
   ```

2. Install dependencies:  
   ```bash
   pip install opencv-python mediapipe numpy
   ```

3. Run the program:  
   ```bash
   python main.py
   ```

---

## **Future Improvements**

- Add more emotion categories like **sad**, **confused**, or **excited**.
- Incorporate machine learning models for improved accuracy.
- Store expression data for historical analysis.
- Enhance performance with GPU acceleration.

---

## **Contributing**

Contributions are welcome! Feel free to fork the repository, open issues, or submit pull requests.

---

## **License**

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## **Demo**

A live demo will be added soon! Stay tuned. 😊

---

### Videos:
- [expression_Demo.mkv](./7_Real_Time_Facial_Expression_Detection_using_Mediapipe_and_OpenCV\expression_Demo.mkv)


## 8_Driver_Drowsiness_and_Distraction_Detection_System

---

# **Driver Drowsiness and Distraction Detection System**

This project uses advanced computer vision techniques to detect driver drowsiness and distractions, helping to enhance road safety by providing timely warnings.

---

## **Table of Contents**
1. [Features](#features)
2. [Technologies Used](#technologies-used)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Functionality](#functionality)
6. [Project Structure](#project-structure)
7. [Demo](#demo)
8. [License](#license)

---

## **Features**
- **Drowsiness Detection**: Tracks eye aspect ratio (EAR) to monitor eye closure and detect signs of drowsiness.
- **Phone Detection**: Identifies if the driver is using a phone, a major source of distraction.
- **Flashing Alert**: Activates a flashing screen for visual alerts when drowsiness or distraction is detected.
- **Voice Warnings**: Uses Text-to-Speech (TTS) for auditory warnings to alert the driver.
- **Real-Time Monitoring**: Processes video feed from the webcam in real time.
- **Configurable Thresholds**: Allows easy tuning of EAR thresholds and alert intervals.

---

## **Technologies Used**
- **Python**: Core programming language.
- **OpenCV**: For video processing and face/eye detection.
- **Mediapipe**: For facial landmarks detection.
- **YOLOv8**: For real-time object detection (phone detection).
- **pyttsx3**: For Text-to-Speech warnings.
- **NumPy**: For numerical operations.
- **Time**: For timing-related logic.

---

## **Installation**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/driver-drowsiness-distraction.git
   cd driver-drowsiness-distraction
   ```

2. **Create and Activate a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 Weights**
   - Download the YOLOv8 weights (`yolov8n.pt`) from the [Ultralytics YOLOv8 repository](https://github.com/ultralytics/ultralytics) and place it in the project directory.

---

## **Usage**

1. **Run the Script**
   ```bash
   python main.py
   ```

2. **How It Works**:
   - A webcam feed will open.
   - The system monitors your eyes for drowsiness and detects phone usage.
   - Alerts (visual flashing and TTS warnings) will be triggered based on predefined thresholds.

3. **Adjust Configurations**:
   - You can modify thresholds and parameters (e.g., EAR threshold, alert interval) in the `main.py` file.

---

## **Functionality**

### 1. **Drowsiness Detection**
   - Uses the Eye Aspect Ratio (EAR) to calculate eye closure.
   - If the eyes remain closed for a continuous period, it triggers a warning.

### 2. **Phone Detection**
   - Uses YOLOv8 for detecting phones in the video feed.
   - If a phone is detected in the frame, it issues a distraction warning.

### 3. **Flashing Screen**
   - Provides a visual alert by toggling a full-screen flash.

### 4. **Text-to-Speech Warnings**
   - Alerts the driver audibly with predefined warning messages.

---

## **Project Structure**

```plaintext
driver-drowsiness-distraction/
├── main.py               # Main script for running the detection system
├── requirements.txt      # Dependencies required for the project
├── README.md             # Project documentation
├── yolov8n.pt            # YOLOv8 model weights



---
```

## **License**
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## **Contributing**
We welcome contributions to enhance the project! Feel free to submit issues or pull requests.

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Submit a pull request with a detailed explanation.

---

If you have any questions or feedback, please feel free to reach out!

--- 

### Videos:
- [implementation.mp4](./8_Driver_Drowsiness_and_Distraction_Detection_System\implementation.mp4)


## 9_DeepLabV3_Image_Segmentation_with_Foreground_Background_Separation_and_Text_Addition

---

# DeepLabV3 Image Segmentation with Foreground-Background Separation and Text Addition

This project demonstrates how to use a pre-trained DeepLabV3 model from the `torchvision` library for semantic segmentation, separating the foreground and background of an image. Additionally, it allows adding custom text to the background, making it useful for various applications such as image editing or creative content generation.

## Features:
- **Semantic Segmentation:** Using a pre-trained DeepLabV3 model with ResNet-101 backbone to segment the image into various categories (e.g., person, animal, background, etc.).
- **Foreground-Background Separation:** The foreground and background of the image are separated based on the segmentation map.
- **Text Overlay:** Customizable text can be added to the background of the image, allowing for further creative manipulation.
- **Visualization:** The original image, segmentation map, foreground, and background with text are displayed using `matplotlib` for easy visualization.

## Requirements:
- Python 3.x
- PyTorch and torchvision
- PIL (Pillow)
- matplotlib
- NumPy

## Installation:
Clone this repository and install the required dependencies:

```bash
git clone https://github.com/yourusername/deeplabv3-image-segmentation.git
cd deeplabv3-image-segmentation
pip install -r requirements.txt
```

## Usage:
1. Make sure to replace the `image_path` variable with the path to your image.
2. Run the script to perform segmentation, separate the foreground and background, and add text to the background.
3. The results (original image, segmentation map, foreground, and background with text) will be displayed using `matplotlib`.

```bash
python segmentation_script.py
```
---

## Usage for Web Application:

1. **Run the Flask Application**: 
   - Ensure you have all the dependencies installed by running the following command:
   ```bash
   pip install -r requirements.txt
   ```
   
2. **Start the Flask App**:
   - Replace the `image_path` and any necessary configurations in the `app.py` file if required.
   - Start the Flask server by running:
   ```bash
   python app.py
   ```
   - This will start a local web server, and you can access the app in your browser by navigating to `http://127.0.0.1:5000/`.

3. **Uploading an Image**:
   - On the web app interface, use the file upload option to select and upload an image.
   - The app will process the image, perform segmentation using the pre-trained DeepLabV3 model, separate the foreground and background, and add text to the background.
   
4. **View Results**:
   - After uploading, the app will display:
     - **Original Image**
     - **Segmentation Map** (color-coded)
     - **Foreground Image**
     - **Background with Text Overlay** (in a separate preview section)
   
5. **Customization**:
   - You can modify the text that is added to the background through the web interface.
   - The output will be displayed dynamically in a preview area.

---

## File Structure:

The project follows a simple structure to keep the code organized:

```
deeplabv3-image-segmentation/
│
├── app.py                  # Flask application for the web interface
├── segmentation_script.py  # Standalone script for running segmentation
├── static/                 # Static folder for serving uploaded images
│   ├── uploads/            # Folder for storing uploaded images temporarily
│   └── results/            # Folder for saving processed results
├── templates/              # HTML templates for rendering the web interface
│   └── index.html          # Main HTML page for the web app
├── requirements.txt        # List of Python dependencies
├── README.md               # Project documentation
└── LICENSE                 # License file
```

### Detailed File Breakdown:
- **`app.py`**: This is the main entry point for the web application. It handles routing, processing of images, and rendering the results on the frontend.
  
- **`segmentation_script.py`**: A standalone script that can be used to run the segmentation process outside of the web app context. Useful for testing or batch processing.
  
- **`static/`**: A folder for serving files such as uploaded images and result images. The `uploads/` folder stores the raw images that are uploaded by the user, and `results/` stores the output images after processing.
  
- **`templates/`**: Contains the HTML files used to render the web interface. The `index.html` file is the primary interface where users can upload images and view the results.

- **`requirements.txt`**: Contains the Python dependencies required to run the project, including Flask, PyTorch, torchvision, and others.
  
- **`README.md`**: This file, which contains the documentation for the project.

---

## Example Output:
- **Original Image**
- **Segmentation Map** (using a jet color map)
- **Foreground Image** (only the segmented foreground)
- **Background with Text** (background with customizable text added)

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments:
- [DeepLabV3 model](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/) from `torchvision`.
- The idea for separating foreground and background is inspired by image segmentation applications in computer vision.

---


## results


## uploads


