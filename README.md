# Car Number Plate Recognition using YOLOv8 and PaddleOCR

This repository implements a Car Number Plate Recognition system using **YOLOv8** for vehicle detection and **PaddleOCR** for optical character recognition (OCR) of the license plates.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Model Details](#model-details)
4. [Installation](#installation)
    1. [Clone the Repository](#1-clone-the-repository)

## Overview

The project is designed to detect cars and extract the number plate information from images using state-of-the-art deep learning models. YOLOv8 is used for vehicle detection, and PaddleOCR is used for optical character recognition on the detected plates.

## Dataset

The dataset used for training is from **Roboflow**. It contains images of cars with labeled number plates.

- The dataset is split into:
  - **Training Set**: 7000+ images
  - **Validation Set**: 2000+ images
  - **Test Set**: 1000+ images

You can access the dataset from **Roboflow** (include a link to the dataset if applicable).

## Model Details

- **Model Used**: YOLOv8 (for object detection)
- **OCR Tool**: PaddleOCR (for extracting text from the detected number plates)

## Installation

Follow the steps below to set up the project on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/Asif556/Car_Number_Plate_Recognition.git

