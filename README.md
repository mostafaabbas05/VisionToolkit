# Vision Toolkit - Image Processing GUI

This is a **PyQt5-based desktop application** for applying various **image processing** techniques using **OpenCV**. The app offers a graphical interface for edge detection, filtering, image segmentation, transformations, and more — all without writing a single line of code!

## 📸 Features

- ✅ Load and display color or grayscale images.
- ✂️ Edge Detection:
  - Prewitt
  - Robert
  - Canny
  - Laplacian of Gaussian (LoG)
- 🎨 Image Segmentation:
  - Histogram-based Thresholding
  - Manual Thresholding
  - Adaptive Thresholding
  - Otsu's Method
- 🧹 Smoothing Filters:
  - Gaussian Blur
  - Mean Filter
  - Median Filter
  - Bilateral Filter
- 🔄 Transformations:
  - Rotation (Slider controlled)
  - Translation (X & Y Sliders)
- 💾 Save processed images to disk.
- 🔁 Reset the application to its initial state.

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3 installed along with the following libraries:

```bash
pip install opencv-python PyQt5 numpy scipy




## Run the App
- Update the UI path in the script (design.ui) to match your system path:
- uic.loadUi(r"C:\Path\To\Your\design.ui", self)

Then run:
python vision_toolkit.py

## 🗂️ Project Structure

### 📁 VisionToolkit
- ┣ 📜 vision_toolkit.py
- ┣ 📄 design.ui         # Qt Designer UI file
- ┣ 🖼️ screenshot.png     # Optional: GUI screenshot




## 💡 Notes

- All filters are adjustable via sliders for real-time updates.
- Make sure `design.ui` exists and matches the widgets expected in the code (e.g. `pushButtons`, `sliders`, `labels`).

## 👨‍💻 Author

Developed by **Mostafa** — part of a practical computer vision project using PyQt5 & OpenCV.

## 📃 License

This project is for educational and non-commercial use. Feel free to use and modify it as needed.
