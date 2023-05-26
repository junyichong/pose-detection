import cv2
import mediapipe as mp
import numpy as np
import csv
import os


# Get coordinates of body part
def get_coordinates(landmarks, body_part):
    if hasattr(mp_pose.PoseLandmark, body_part):
        body_part_index = mp_pose.PoseLandmark[body_part].value
        x = landmarks[body_part_index].x
        y = landmarks[body_part_index].y
        z = landmarks[body_part_index].z
        return [x, y, z]
    else:
        return None


# Calculate angle between three points
def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

video_path = 'test_videos/type3.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

folder_path = os.path.join(os.getcwd(), 'results')
os.makedirs(folder_path, exist_ok=True)
csv_path = os.path.join(folder_path, 'type3.csv')

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['left angle', 'right angle'])


with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Extract landmarks
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates
            left_shoulder = get_coordinates(landmarks, 'LEFT_SHOULDER')
            left_elbow = get_coordinates(landmarks, 'LEFT_ELBOW')
            left_wrist = get_coordinates(landmarks, 'LEFT_WRIST')

            print(left_shoulder)

            right_shoulder = get_coordinates(landmarks, 'RIGHT_SHOULDER')
            right_elbow = get_coordinates(landmarks, 'RIGHT_ELBOW')
            right_wrist = get_coordinates(landmarks, 'RIGHT_WRIST')

            print(right_shoulder)

            # Calculate angle
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

            # Visualize angle
            cv2.putText(image, str(left_angle), 
                           tuple(np.multiply(left_elbow, [frame_width, frame_height]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                                )
            cv2.putText(image, str(right_angle), 
                           tuple(np.multiply(right_elbow, [frame_width, frame_height]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
                                )

        except:
            pass
        
        # Write results to .csv file
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([left_angle, right_angle])

        # Render detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        cv2.imshow('Pose Detection Result', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
