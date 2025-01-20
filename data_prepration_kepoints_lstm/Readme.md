# Data Preparation for Action Recognition in Sports

This repository contains tools and scripts for preparing data to train action recognition models in sports. The data preparation pipeline ensures standardization of input data, regardless of how it was initially annotated or formatted by the client.

## Overview

The final processed data format is designed for action recognition tasks, specifically utilizing temporal keypoints for a person. This format allows for training recurrent neural networks (RNNs) to learn and predict actions based on sequences of keypoints.

## Features

- **Keypoint Extraction**: Uses [Mediapipe](https://ai.google.dev/edge/mediapipe/solutions/guide) to extract keypoints of the person of interest in video frames.
- **Standardized Output**: Keypoints from frames of interest are saved in CSV files for each video. These CSVs include:
  - Frame numbers
  - Keypoints
  - Labels for the action being performed in the frame
- **Flexible Input**: Can be used by changing minimal logic for various client-provided formats and annotations, standardizing them for preprocessing.

## Workflow

1. **Input Videos and Annotations**:
   - Provide input video files and their corresponding annotations (if available).
   - The input format may vary based on client specifications(for which the scripts will need to be adjusted).

2. **Keypoint Extraction**:
   - Mediapipe extracts temporal keypoints of the person of interest for each frame.

3. **Output Generation**:
   - Keypoints, along with frame numbers and action labels, are stored in CSV files.

4. **Usage**:
   - These CSV files serve as the input for training action recognition models, such as RNNs.

## Applications

The processed data can be used in various applications, including but not limited to:
- Sports analytics
- Player performance evaluation
- Automated video highlights

## Requirements

- Python 3.x
- Mediapipe
- Other dependencies (listed in `requirements.txt`)

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo.git
