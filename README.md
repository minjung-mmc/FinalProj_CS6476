# FinalProj_CS6476

## Final Project for Georgia Tech CS 6476 ? Computer Vision
## Digit Detection & Recognition Pipeline (MSER + CNN)

## Project Overview

This project implements a two-stage pipeline to detect and recognize digit sequences in natural images such as street-view house numbers.

Region Proposal

MSER (Maximally Stable Extremal Regions)

Multi-scale image pyramid

Digit Classification

Custom CNN

or Fine-tuned VGG16

Final output: 5 processed images with bounding boxes + predicted sequences saved to graded_images/.

## Repository Structure
'''
FinalProj_CS6476/
戍式式 checkpoints/        # trained model weights (.pth)
戍式式 run.py              # main entry script
戍式式 network.py          # CNN architectures (CustomCNN, VGG16 finetune)
戍式式 utils.py            # MSER, NMS, pyramid, preprocessing
戍式式 config.py           # configuration system
戍式式 inputs/             # input demo test images
戍式式 graded_images/      # output images written during test mode
戌式式 README.md
'''

? Requirements

manually:

torch
torchvision
opencv-python
numpy
Pillow

## Ⅱ? How to Run the Project
1 Run Inference (Digit Detection + Recognition)

Produces 5 output images in graded_images/.

Custom CNN
python run.py --mode test --model_type CustomCNN


This will:

Load checkpoint from checkpoints/

Run MSER + image pyramid region proposal

Classify each ROI

Save results into graded_images/1.png to 5.png

