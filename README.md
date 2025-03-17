
# 🖼️ YOLO Object Detection with Gradio

This project is a **YOLO-based object detection** application using **Gradio** for an interactive UI. It enables real-time detection in images and videos with adjustable settings for better accuracy and performance.

---

## 🚀 Features
✅ Real-time object detection using YOLOv8  
✅ Supports multiple models: `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`  
✅ Adjustable image size and confidence threshold  
✅ Display annotated results instantly  
✅ Fast and lightweight models for low-latency detection  

---

## 🏗️ Project Structure
```plaintext
object-Detection/
├── app.py             # Main Gradio app file
├── requirements.txt   # List of dependencies
├── README.md          # Project documentation
├── data/              # Sample images and videos
└── models/            # Pre-trained YOLO models
```

---

## 🛠️ Installation
1. **Clone the Repository**  
```bash
git clone https://github.com/Gowthamhegde/object-Detction.git
cd object-Detction
```

2. **Create a Virtual Environment**  
```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
.env\Scriptsctivate    # Windows
```

3. **Install Dependencies**  
```bash
pip install -r requirements.txt
```

---

## 🎯 How to Run the Project
1. **Start the App**  
```bash
python app.py
```

2. **Access in Browser**  
Open [http://127.0.0.1:7860](http://127.0.0.1:7860)  

3. **Upload an Image/Video**  
- Upload any image or video.  
- Choose the YOLO model type and adjust the confidence threshold.  
- Click **"Submit"** to view results.  

---

## 📋 Requirements
- Python 3.8+  
- Libraries:  
  - `gradio`  
  - `opencv-python-headless`  
  - `ultralytics`  

---

## 🧠 How It Works
1. The uploaded image/video is passed to the selected YOLO model.  
2. The model processes the input, detects objects, and generates bounding boxes.  
3. Results are rendered in real-time using Gradio's interface.  

---

## 🎥 Demo  
![Demo](./Screenshot%202025-03-14%20195307.png)

---

## 📌 Notes  
- Ensure your Python environment is correctly configured.  
- Higher confidence thresholds reduce false positives but may miss some objects.  
- Try different YOLO models for balancing speed and accuracy.  

---

## 🤝 Contributing
Feel free to open an issue or submit a pull request! 😎  
