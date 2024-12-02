import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Indices for iris landmarks and eye corners
left_iris_indices = [474, 475, 476, 477]
right_iris_indices = [469, 470, 471, 472]
left_eye_inner = 362
left_eye_outer = 263
right_eye_inner = 133
right_eye_outer = 33

def get_iris_position(face_landmarks, iris_indices, frame_width, frame_height):
    iris_x = np.mean([face_landmarks.landmark[i].x for i in iris_indices]) * frame_width
    iris_y = np.mean([face_landmarks.landmark[i].y for i in iris_indices]) * frame_height
    return (iris_x, iris_y)

def calculate_relative_position(iris_pos, eye_inner_pos, eye_outer_pos):
    eye_width = eye_outer_pos[0] - eye_inner_pos[0]
    iris_offset_x = iris_pos[0] - eye_inner_pos[0]
    iris_relative_x = iris_offset_x / eye_width  # 0.5 means looking straight
    return iris_relative_x

def get_head_pose(face_landmarks, frame_width, frame_height, camera_matrix, dist_coeffs):
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye corner
        (225.0, 170.0, -135.0),  # Right eye corner
        (-150.0, -150.0, -125.0),  # Left mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    image_points = np.array([
        (face_landmarks.landmark[1].x * frame_width, face_landmarks.landmark[1].y * frame_height),  # Nose tip
        (face_landmarks.landmark[152].x * frame_width, face_landmarks.landmark[152].y * frame_height),  # Chin
        (face_landmarks.landmark[33].x * frame_width, face_landmarks.landmark[33].y * frame_height),  # Left eye corner
        (face_landmarks.landmark[263].x * frame_width, face_landmarks.landmark[263].y * frame_height),  # Right eye corner
        (face_landmarks.landmark[61].x * frame_width, face_landmarks.landmark[61].y * frame_height),  # Left mouth corner
        (face_landmarks.landmark[291].x * frame_width, face_landmarks.landmark[291].y * frame_height)  # Right mouth corner
    ], dtype='float64')

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs)
    
    # Get rotation matrix and euler angles
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]
    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
    pitch, yaw, roll = eulerAngles  # Extract head orientation angles
    
    return pitch, yaw, roll

def process_frame(image, face_mesh, camera_matrix, dist_coeffs):
    frame_height, frame_width, _ = image.shape
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            left_iris_pos = get_iris_position(face_landmarks, left_iris_indices, frame_width, frame_height)
            right_iris_pos = get_iris_position(face_landmarks, right_iris_indices, frame_width, frame_height)

            left_eye_inner_pos = (face_landmarks.landmark[left_eye_inner].x * frame_width,
                                  face_landmarks.landmark[left_eye_inner].y * frame_height)
            left_eye_outer_pos = (face_landmarks.landmark[left_eye_outer].x * frame_width,
                                  face_landmarks.landmark[left_eye_outer].y * frame_height)
            right_eye_inner_pos = (face_landmarks.landmark[right_eye_inner].x * frame_width,
                                   face_landmarks.landmark[right_eye_inner].y * frame_height)
            right_eye_outer_pos = (face_landmarks.landmark[right_eye_outer].x * frame_width,
                                   face_landmarks.landmark[right_eye_outer].y * frame_height)

            left_iris_relative_x = calculate_relative_position(left_iris_pos, left_eye_inner_pos, left_eye_outer_pos)
            right_iris_relative_x = calculate_relative_position(right_iris_pos, right_eye_inner_pos, right_eye_outer_pos)

            pitch, yaw, roll = get_head_pose(face_landmarks, frame_width, frame_height, camera_matrix, dist_coeffs)

            horizontal_adjustment = 0.1 * abs(yaw / 70)
            if (0.4 - horizontal_adjustment/2) < left_iris_relative_x < (0.6 + horizontal_adjustment/2) and \
               (0.4 - horizontal_adjustment) < right_iris_relative_x < (0.6 + horizontal_adjustment):
                return "Concentrated"
            else:
                return "Not Concentrated"
    return "No face detected"
