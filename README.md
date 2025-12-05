# FinalProj_CS6476

## Final Project for Georgia Tech CS 6476  Computer Vision
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
```
FinalProj_CS6476/
|--checkpoints/        # trained model weights (.pth)
|-- run.py              # main entry script
|-- network.py          # CNN architectures (CustomCNN, VGG16 finetune)
|-- utils.py            # MSER, NMS, pyramid, preprocessing
|-- config.py           # configuration system
|-- inputs/             # input demo test images
|-- graded_images/      # output images written during test mode
|-- README.md
```

## Requirements

manually:

python == 3.11
torch == 2.0.1
torchvision == 0.15.2
opencv-python
numpy
Pillow

## How to Run the Project
### Run Inference (Digit Detection + Recognition)

Produces 5 output images in graded_images/.
```
python run.py --mode test --model_type CustomCNN
```

This will:

Load checkpoint from checkpoints/

Run MSER + image pyramid region proposal

Classify each ROI

Save results into graded_images/1.png to 5.png

