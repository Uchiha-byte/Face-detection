import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
from datetime import datetime
import time  # Import time module for delay

# Load known faces and their names
known_face_encodings = []
known_face_names = []

# Load images from the "known_faces" folder and encode them
def load_known_faces():
    folder_path = 'known_faces'
    for file_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, file_name)
        print(f"Loading image: {image_path}")
        image = face_recognition.load_image_file(image_path)
        
        # Ensure there is at least one face in the image
        encodings = face_recognition.face_encodings(image)
        if encodings:
            encoding = encodings[0]  # Get the encoding for the first face detected
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(file_name)[0])  # Use the file name (without extension) as the name
        else:
            print(f"No faces found in {image_path}")

# Mark attendance in a CSV file
def mark_attendance(name):
    file_name = 'attendance.csv'
    now = datetime.now()
    time_string = now.strftime('%H:%M:%S')
    date_string = now.strftime('%Y-%m-%d')

    # Check if the CSV file exists and create if not
    if not os.path.exists(file_name):
        df = pd.DataFrame(columns=['Name', 'Date', 'Time'])
        df.to_csv(file_name, index=False)

    # Avoid duplicate entries
    df = pd.read_csv(file_name)
    if not ((df['Name'] == name) & (df['Date'] == date_string)).any():
        with open(file_name, 'a') as f:
            f.write(f'\n{name},{date_string},{time_string}')
            print(f"Attendance logged for {name} at {time_string}")

# Main function to recognize faces and mark attendance
def recognize_faces():
    video_capture = cv2.VideoCapture(0)
    last_face_seen = None  # Keep track of the last face detected
    last_logged_name = None  # To store last logged name to avoid duplicates in quick succession

    while True:
        ret, frame = video_capture.read()
        
        # Resize the frame for faster processing and convert it to RGB (OpenCV loads images as BGR)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

        # Find face locations in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)

        # Debug: Show how many faces were found
        print(f"Found {len(face_locations)} face(s) in the frame.")

        # If faces are found, encode them
        if len(face_locations) > 0:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            # Process each face
            for face_encoding, face_location in zip(face_encodings, face_locations):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"  # Default name if no match is found

                # If a match is found, get the name of the person
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]
                    
                    # Only log attendance if it's a new face or different from the last logged one
                    if name != last_logged_name:
                        mark_attendance(name)
                        last_logged_name = name  # Update last logged name

                # Draw the face box and label the name on the video feed
                top, right, bottom, left = face_location
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Display the video feed
        cv2.imshow('Video', frame)

        # Add delay to prevent excessive recognition attempts and make it less "jumpy"
        time.sleep(1)  # Adding a 1-second delay between frames to avoid quick re-recognition of faces

        # Press 'q' to quit the video feed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

# Load known faces and start recognition
load_known_faces()
recognize_faces()
