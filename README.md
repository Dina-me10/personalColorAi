# Personal Color AI

Takes a selfie, finds the face, measures cheek skin color in CIELAB, and decides your personal color season (Spring Warm, Summer Cool, Autumn Warm, Winter Cool).

## How to run

```bash
python3 -m venv venv
source venv/bin/activate
pip install opencv-python numpy
```

Put your own photo in this folder as `selfie.jpg`, then run the steps one by one:

```bash
python step1.py   # open the image, print size and center pixel color (BGR)
python step2.py   # detect the face with YuNet -> step2_face.jpg
python step3.py   # find both cheeks from face landmarks -> step3_cheeks.jpg
python step4.py   # measure cheek color: L*, a*, b*, hue angle
```

The face detection model `face_detection_yunet_2023mar.onnx` is from [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) (MIT license).
