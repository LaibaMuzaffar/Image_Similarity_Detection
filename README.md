#  AI Image Similarity Detection System

An intelligent **AI-powered Image Similarity Detection System** built with **Python**, **Computer Vision**, and **Deep Learning**.

The project automatically compares two images and determines how similar they are by combining **Face Recognition**, **Vision Transformers (CLIP)**, and **Perceptual Hashing**.

Unlike traditional image comparison methods that only compare pixels, this system understands the **semantic meaning** of images. It automatically detects whether the images contain faces and selects the most appropriate AI model for comparison.

The application is completely desktop-based and requires **no HTML, CSS, JavaScript, or web browser**. Everything runs directly in Python.

---

#  Features

* AI-powered image similarity detection
* Automatic face detection
* Face comparison using InsightFace embeddings
* General object comparison using OpenAI CLIP
* Perceptual Hash (pHash) structural similarity
* Hybrid similarity scoring
* Modern Tkinter desktop GUI
* GPU support (CUDA) if available
* CPU fallback support
* Automatic desktop image loading
* Professional similarity visualization

---

#  AI Models Used

## 1. InsightFace (Face Recognition)

Used whenever both images contain human faces.

Features:

* Face Detection
* Face Alignment
* Face Embeddings
* Cosine Similarity Matching

---

## 2. OpenAI CLIP

Used when faces are not detected.

CLIP understands:

* Objects
* Animals
* Vehicles
* Landscapes
* Buildings
* Products
* General image semantics

Model Used:

```
ViT-B/32
```

---

## 3. Perceptual Hash (pHash)

Provides structural similarity between two images and is combined with CLIP to improve the final similarity score.

---

# 🛠 Technologies Used

* Python 3.10+
* PyTorch
* OpenAI CLIP
* InsightFace
* ONNX Runtime
* Pillow
* NumPy
* ImageHash
* Tkinter

---

#  Project Structure

```text
Image-Similarity-System/
│
├── compare_images.py
├── requirements.txt
├── README.md
└── sample_images/
      ├── image1.jpg
      └── image2.jpg
```

---

#  System Requirements

* Windows 10 / Windows 11
* Python 3.10 (Recommended)
* Anaconda (Recommended)
* Internet connection (only for first model download)
* Minimum 8 GB RAM
* NVIDIA GPU (Optional but recommended)

---

#  Installation Guide

## Step 1 — Install Anaconda

Download and install **Anaconda Individual Edition** from:

https://www.anaconda.com/products/distribution

After installation, open:

**Anaconda Prompt**

---

## Step 2 — Create a New Environment

Run:

```bash
conda create -n image_similarity python=3.10
```

---

## Step 3 — Activate the Environment

```bash
conda activate image_similarity
```

You should now see:

```text
(image_similarity) C:\Users\YourName>
```

This means your environment is active.

---

## Step 4 — Open VS Code

Navigate to your project folder:

```bash
cd path\to\Image-Similarity-System
```

Then open VS Code:

```bash
code .
```

If the `code` command is not recognized, open VS Code manually and select **File → Open Folder**.

---

#  Install Required Libraries

Run each command inside the activated Anaconda environment.

## Install PyTorch

CPU Version

```bash
pip install torch torchvision torchaudio
```

If you have an NVIDIA GPU with CUDA support, install the CUDA version from the official PyTorch website:

https://pytorch.org/get-started/locally/

---

## Install OpenAI CLIP

```bash
pip install ftfy regex tqdm
```

```bash
pip install git+https://github.com/openai/CLIP.git
```

---

## Install InsightFace

```bash
pip install insightface
```

---

## Install ONNX Runtime

CPU Version

```bash
pip install onnxruntime
```

GPU Version (Optional)

```bash
pip install onnxruntime-gpu
```

---

## Install Remaining Libraries

```bash
pip install numpy pillow imagehash
```

---

## Tkinter

Tkinter is included with most Python installations.

To verify:

```bash
python -m tkinter
```

If a small demo window opens, Tkinter is installed correctly.

---

#  Installing Everything at Once

Create a `requirements.txt` file:

```text
numpy
torch
torchvision
torchaudio
pillow
imagehash
insightface
onnxruntime
ftfy
regex
tqdm
git+https://github.com/openai/CLIP.git
```

Then install everything using:

```bash
pip install -r requirements.txt
```

---

#  First-Time Model Download

When you run the project for the first time:

* CLIP automatically downloads the **ViT-B/32** model.
* InsightFace automatically downloads the **buffalo_l** face recognition model.

These models are stored locally and will not be downloaded again unless deleted.

---

# Running the Project

Make sure your environment is activated:

```bash
conda activate image_similarity
```

Run the project:

```bash
python compare_images.py
```

---

#  How to Compare Images

Place two images in the location expected by the script (or update the image paths in the code).

Run:

```bash
python compare_images.py
```

The program will:

1. Load the AI models.
2. Detect faces in both images.
3. Compare faces if present.
4. Otherwise compare semantic image content using CLIP.
5. Calculate structural similarity using Perceptual Hash.
6. Combine the results into a final similarity score.
7. Display the results in a modern desktop GUI.

---

#  Similarity Pipeline

```text
Image 1          Image 2
     │               │
     └──────┬────────┘
            │
      Face Detection
            │
      ┌─────┴─────┐
      │           │
 Faces Found?     No Faces
      │           │
InsightFace      OpenAI CLIP
      │           │
      └─────┬─────┘
            │
    Perceptual Hash
            │
     Hybrid Scoring
            │
     Similarity %
            │
 Desktop GUI Result
```

---

#  Similarity Interpretation

| Similarity | Meaning                 |
| ---------- | ----------------------- |
| 90–100%    | Nearly identical images |
| 75–89%     | Very similar            |
| 50–74%     | Moderately similar      |
| 25–49%     | Slight similarity       |
| 0–24%      | Different images        |

---

#  Performance

* Face comparison is extremely fast.
* CLIP provides semantic understanding instead of simple pixel comparison.
* GPU acceleration is used automatically if CUDA is available.

---

#  Future Improvements

* Drag-and-drop image selection
* Batch image comparison
* Folder-to-folder similarity search
* Duplicate image finder
* Video frame similarity detection
* PDF image comparison
* Image retrieval system
* Web application using Flask or FastAPI
* Docker deployment
* REST API support

---

#  Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

#  License

This project is released under the **MIT License**, allowing free use, modification, and distribution for educational and personal purposes.

---

#  Author

Developed as a Computer Vision and Deep Learning project using **InsightFace**, **OpenAI CLIP**, and **Perceptual Hashing** to provide intelligent image similarity detection with a standalone Python desktop interface.
