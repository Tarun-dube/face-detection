import cv2
import math
import argparse



def highlightFace(net, frame, conf_threshold=0.7):
    """
    Detect faces in the frame using the provided neural network.
    Draw rectangles around detected faces.

    Parameters:
    - net: The neural network for face detection.
    - frame: The image frame in which to detect faces.
    - conf_threshold: The confidence threshold for detecting faces.

    Returns:
    - frameOpencvDnn: The frame with rectangles drawn around detected faces.
    - faceBoxes: A list of bounding boxes for the detected faces.
    """
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    # Create a blob from the image
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    
    # Set the input for the neural network
    net.setInput(blob)
    # Perform face detection
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            # Calculate the coordinates of the bounding box
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            # Draw a rectangle around the detected face
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    
    return frameOpencvDnn, faceBoxes



# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()

# Paths to the model files
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Mean values for the models
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)

# Age and gender lists
ageList = ['(1-5) Baby Child', '(6-10) Children', '(11-15) Infant', '(16-20) Teenager', '(21-25) Adult', '(25-35) Men or Women', '(36-53) Men or Women', '(54-100) Aged']
genderList = ['Male', 'Female']

# Load the models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Capture video from the specified image file or webcam
video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break

    # Detect faces in the frame
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")

    for faceBox in faceBoxes:
        # Extract the face from the frame
        face = frame[max(0, faceBox[1] - padding):
                     min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):
                     min(faceBox[2] + padding, frame.shape[1] - 1)]

        # Create a blob from the face image
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
        # Predict gender
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]
        print(f'Gender: {gender}')

        # Predict age
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]
        print(f'Age: {age[1:-1]} years')

        # Put text with gender and age on the result image
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)
        break
