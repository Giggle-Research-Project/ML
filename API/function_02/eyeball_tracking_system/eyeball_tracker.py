import mediapipe as mp

# Indices for key landmarks on the iris and eye corners
left_iris_index = 468
left_eye_inner = 133
left_eye_outer = 33
right_iris_index = 473
right_eye_inner = 362
right_eye_outer = 263

# Initialize MediaPipe Face Mesh
def instantiate_faceMesh():
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        static_image_mode=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return face_mesh


# Function to calculate gaze direction
def is_concentrated(face_landmarks, frame_width, frame_height):
    
    # Get positions of left and right iris centers
   
    left_rye_z = face_landmarks.landmark[left_iris_index].z
    right_iris_z = face_landmarks.landmark[right_iris_index].z
    
    left_eye_y = face_landmarks.landmark[left_eye_outer].y * frame_height
    right_eye_y = face_landmarks.landmark[right_eye_outer].y * frame_height
    eye_y_difference = abs(left_eye_y - right_eye_y)
    
    # Tilted face
    if eye_y_difference > 15: # higher = more angle
        return False
    
    left_iris_x = face_landmarks.landmark[left_iris_index].x * frame_width
    right_iris_x = face_landmarks.landmark[right_iris_index].x * frame_width

    # Get eye corner positions
    left_eye_inner_x = face_landmarks.landmark[left_eye_inner].x * frame_width
    left_eye_outer_x = face_landmarks.landmark[left_eye_outer].x * frame_width
    right_eye_inner_x = face_landmarks.landmark[right_eye_inner].x * frame_width
    right_eye_outer_x = face_landmarks.landmark[right_eye_outer].x * frame_width

    # Calculate relative positions for gaze detection
    left_gaze_ratio = (left_iris_x - left_eye_inner_x) / (left_eye_outer_x - left_eye_inner_x)
    right_gaze_ratio = (right_iris_x - right_eye_inner_x) / (right_eye_outer_x - right_eye_inner_x)
    
    z_diff = abs(left_rye_z - right_iris_z)
    
    # Check the head pose
    if z_diff > 0.028: # head pose (higher = more angle)
        
        if (left_rye_z > right_iris_z): # looking left

            if  left_gaze_ratio > 0.55: # higher = more angle
                 return False
            else:
              return True
            
        else: # looking right
            if  right_gaze_ratio > 0.55: # higher = more angle
                return False
            else:
              return True
             
    # Check if both irises are centered within each eye
    if abs(left_gaze_ratio - right_gaze_ratio) < 0.14: # higher = more angle
        return True

    return False