# yolo-model
This project uses a custom-trained YOLO object detection model to detect and identify various types of candy in real time through a webcam. Once candies are identified, the script calculates the total calories and sugar content based on preloaded nutritional information.

## Features

- Real-time object detection with YOLOv8
- Displays bounding boxes, labels, and confidence scores
- Calculates total calories and sugar in the frame
- Supports pausing and image capture
- Option to record output video

## Candies Recognized

- Nerds
- Hi-Chew
- Lollipop
- Starburst

## Nutrition Info
Each candy has predefined calorie and sugar values (from Costco packaging) stored in a dictionary like this:
nutrition_info = {
    'starburst': [40, 6.6],
    'lollipop': [20, 4],
    'hichew': [21, 3.5],
    'nerds': [70, 20]
}

## How It Works
- The webcam captures a live video feed.
- A YOLO11 model detects candies and labels them.
- Detected candy types are tallied.
- Calories and sugar grams are summed and displayed.

## Requirements
- Python 3.8+
- Ultralytics YOLO
- OpenCV
- Anaconda environment
- Label Studio
- NumPy

## Training the Model
- Take around 100-200 jpgs of your sample dataset, in my case it was candies (nerds, hi-chew, lollipop, starburst)
- Make sure these pictures are from multiple angles, perspectives, lighting.
- You can also use a pre-made dataset from Roboflow, Kaggle, or Open Images V7 (Google)
- Download Anaconda and make a virtual environment
- Install and start Label Studio with: pip install label-studio AND label-studio start.
- Follow the Labeling Data instructions below.
- Use Edje Electronics Google Colab to upload and train the model. https://colab.research.google.com/github/EdjeElectronics/Train-and-Deploy-YOLO-Models/blob/main/Train_YOLO_Models.ipynb#scrollTo=gzaJQ2sGEPhP
- Run the model with python yolo_video.py --model my_model.pt --source usb0 --resolution 1280x720

## Labeling Data
- Once Label Studio opens, creatye an account and click create project.
- Import your images under the Data Import tab. (only import 100 images at a time)
- Click Labeling Setup and delete the premade labels. Add your own based on your data (I did nerds, hi-chew, lollipop, starburst).
- Start labeling by clicking the first image and clicking and dragging boxes around each corresponding object with the correct label.
- Once you're done labeling, export with "YOLO with images".
- Make sure you put data.zip in the correct folder.

## Run script
- python yolo_video.py --model my_model.pt --source usb0 --resolution 1280x720
- python stats.py --model my_model.pt --source usb0 --resolution 1280x720
