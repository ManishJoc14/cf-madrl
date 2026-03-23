
cf-madrl vehicle - v5 2026-03-21 5:42pm
==============================

This dataset was exported via roboflow.com on March 21, 2026 at 12:57 PM GMT

Roboflow is an end-to-end computer vision platform that helps you
* collaborate with your team on computer vision projects
* collect & organize images
* understand and search unstructured image data
* annotate, and create datasets
* export, train, and deploy computer vision models
* use active learning to improve your dataset over time

For state of the art Computer Vision training notebooks you can use with this dataset,
visit https://github.com/roboflow/notebooks

To find over 100k other datasets and pre-trained models, visit https://universe.roboflow.com

The dataset includes 43 images.
Vehicle are annotated in YOLO26 format.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 640x640 (Stretch)

The following augmentation was applied to create 2 versions of each source image:
* 50% probability of horizontal flip
* Randomly crop between 0 and 20 percent of the image
* Random shear of between -15° to +15° horizontally and -15° to +15° vertically
* Random brigthness adjustment of between -25 and +25 percent
* Random Gaussian blur of between 0 and 2 pixels


