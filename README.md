# From Sketch to Space: Interactive 3D Environments from 2D Floor Plans

## Problem Statement

This project aims to construct a computational pipeline that converts 2D architectural floor plans into interactive 3D environments using Unity. The pipeline begins with raster image input, processed using OpenCV to recognize walls, doors, and windows. Extracted vectors are algorithmically converted into simplified 3D geometry, which is then imported into Unity's engine to enable spatial walkthroughs.

The objective is to enable designers and architecture students to quickly visualize their spatial ideas in a realistic setting, enhancing early-stage decision making.

## Motivation

The architectural workflow has long depended on 2D representations, which—though essential—can often be static, ambiguous, or disconnected from spatial intuition. By enabling users to walk through their sketches, this tool introduces an **experiential dimension** to ideation.

## Architectural and Design Relevance

This work connects deeply to architecture's core practice of spatial representation. Just as hand-drafted plans once revolutionized architectural communication, AI-enhanced representations now promise to redefine how we **ideate, prototype, and share space**.

The pipeline supports:
- **Design education** through intuitive, real-time feedback loops  
- **Human-centered design** by making spatial ideation more accessible  
- Future integration with **AR/VR** and **real-time collaboration tools**

## Tech Stack

| Layer          | Technology                          | Description |
|----------------|--------------------------------------|-------------|
| **Frontend**   | **Three.js**                         | Renders extracted geometry as rotatable 3D models directly in the browser |
|                | HTML/CSS/JavaScript                  | Light UI wrapper and interactions |
| **Backend**    | **FastAPI**                          | Serves REST endpoints to receive sketch uploads and return geometry data (JSON or .3dm) |
|                | **Python + OpenCV**                  | Handles preprocessing, edge detection, and vector extraction |
|                | **scikit-image / RDP / NumPy**       | Geometry cleanup, segmentation, and path simplification |
| **3D Engine**  | **Unity**                            | Hosts the real-time first-person walkthrough with lighting and collision |
| **Export**     | `.3dm`, `.obj`, `.json`              | Supports interoperability between browser viewer and Unity walkthrough |
| **3D Geometry**   | **Rhino/Grasshopper** | For advanced geometry processing and interoperability |

## Algorithmic Overview

### 1. Ramer–Douglas–Peucker (RDP) Algorithm

**Purpose**: Simplifies polylines by recursively removing points that deviate the least from a straight path, preserving shape while reducing complexity.

- **Strengths**: Excellent for removing noise from complex contours.  
- **Limitations**: May oversimplify curves if not finely tuned.

**Screenshot: RDP-Simplified Floor Plan**  
![RDP Screenshot](screenshots/rdp_example.jpg)

### 2. Canny Edge Detection

**Purpose**: Detects edges based on image gradients using a multi-stage algorithm (smoothing, gradient calculation, non-maximum suppression, hysteresis thresholding).

- **Strengths**: Good at finding clean, sharp contours in raster input.  
- **Limitations**: Sensitive to noise; may produce fragmented outputs in sketchy drawings.

**Screenshot: Canny Edge Detection Output**  
![Canny Screenshot](screenshots/canny_example.jpg)

### 3. Corner Detection (Harris, Shi-Tomasi)

**Purpose**: Identifies significant "turns" or junctions in geometry, helping to segment walls or capture key vertices.

- **Strengths**: Helps segment paths and distinguish straight vs. curved walls.  
- **Limitations**: Often returns false positives in noisy drawings; pairing with line masks helps improve accuracy.

**Screenshot: Corner Detection Overlay**  
![Corner Screenshot](screenshots/corner_example.jpg)

## Modular Pipeline Design

The pipeline is implemented as a **modular system**, allowing plug-and-play experimentation with different vectorization strategies. This enables comparison between different outputs and supports adaptation to varying sketch qualities.

Modules include:

- Preprocessing (binarization, skeletonization, smoothing)  
- Contour extraction  
- Polyline simplification and cleanup  
- Post-processing (snapping, deduplication, merging)  
- 3D extrusion and export to Unity  
- Browser interface (via **Three.js**) for 3DM rotation and walkthrough launch  

## Demo & Interface

- 🧭 **Unity Walkthrough Demo**: [Link or placeholder]  
- 🌐 **Three.js Viewer**: View and rotate the processed 3DM file in-browser  
- 🗺️ **Side-by-side comparisons**: Raw sketch, processed vectors, and 3D model  

*Screenshots, video walkthroughs, and GIFs go here.*

## Future Work

- Real-time 2D-to-3D conversion via camera or touch interface  
- AR-enhanced sketch input (draw and walk instantly)  
- Shared collaboration space for architects and clients  
- Machine learning-driven symbol detection (e.g., for windows, furniture)

## References

1. Huang, J., Zhang, C., Xiang, S., & Li, Z. (2023). **Plan2Scene: Learning to Generate 3D Interior Scene from 2D Floor Plan**. [arXiv:2306.06441](https://doi.org/10.48550/arXiv.2306.06441)

2. Nguyen, T., Xiang, S., & Zhang, C. (2021). **Plan2Vec: Unsupervised Representation Learning for Floorplan Analysis**. [arXiv:2110.04830](https://doi.org/10.48550/arXiv.2110.04830)

3. Liu, C., Lin, T.-Y., Hua, W., et al. (2020). **Image Matching via Invariant Feature Learning**. [arXiv:2003.05471](https://arxiv.org/abs/2003.05471)

4. Lin, C.-H., Yumer, E., Wang, O., et al. (2020). **Drafting Annotator: Visual Recovery of Floor Plans**. [arXiv:2003.14034](https://arxiv.org/abs/2003.14034)