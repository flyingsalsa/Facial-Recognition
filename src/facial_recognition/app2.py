import cv2
import insightface
import numpy as np
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image # For loading reference image if needed

# Initialize face analysis model
# Use 'CUDAExecutionProvider' for GPU if available and desired
try:
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])
except Exception as e:
    print(f"CUDA not available or error during init with CUDA: {e}. Falling back to CPU.")
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])

app.prepare(ctx_id=0 if 'CUDAExecutionProvider' in app.providers else -1) # ctx_id=0 for GPU, -1 for CPU

def get_face_object_and_embedding(img_data):
    """
    Detects faces and returns the first detected Face object (containing bbox, embedding, etc.).
    Raises ValueError if no faces are detected.
    """
    if img_data is None:
        raise ValueError("Input image data is None")

    faces = app.get(img_data) # Pass the NumPy array directly

    if not faces: # More Pythonic check for empty list
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected in the current frame. Using the first one.")

    return faces[0] # Return the whole Face object

def get_embedding_from_file(image_path):
    """Extract face embedding from an image file path."""
    # Using ins_get_image is a good practice for insightface,
    # as it handles various image formats and color conversions.
    # However, cv2.imread is also fine if you ensure BGR format.
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    face_obj = get_face_object_and_embedding(img) # Re-use the function
    return face_obj.embedding


def compare_faces(emb1, emb2, threshold=0.6): # Adjusted threshold slightly, tune as needed
    """Compare two embeddings using cosine similarity."""
    emb1 = np.asarray(emb1)
    emb2 = np.asarray(emb2)

    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity, similarity > threshold

# --- Configuration ---
WEBCAM_INDEX = 0 # Try 0, 1, 2, etc. if default doesn't work
REFERENCE_IMAGE_PATH = "Faces/face2.jpg" # Path to your reference face image
SIMILARITY_THRESHOLD = 0.5 # Adjust based on your model and use case (buffalo_l often needs lower than 0.65 for "same")

# --- Load reference embedding once ---
try:
    print(f"Loading reference embedding from: {REFERENCE_IMAGE_PATH}")
    reference_embedding = get_embedding_from_file(REFERENCE_IMAGE_PATH)
    print("Reference embedding loaded successfully.")
except Exception as e:
    print(f"Error loading reference image/embedding: {e}")
    print("Please ensure the path is correct and the image contains a detectable face.")
    exit()

# --- Setup OpenCV capture device (webcam) ---
cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print(f"Error: Could not open webcam at index {WEBCAM_INDEX}.")
    exit()

print(f"Webcam {WEBCAM_INDEX} started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break

    display_frame = frame.copy() # Work on a copy to draw on

    try:
        # Get face object (which includes bbox and embedding) from the current webcam frame
        webcam_face_obj = get_face_object_and_embedding(frame) # Use the original frame for detection
        webcam_embedding = webcam_face_obj.embedding
        bbox = webcam_face_obj.bbox.astype(int) # Bounding box [x1, y1, x2, y2]

        # Compare faces
        similarity_score, is_same_person = compare_faces(webcam_embedding, reference_embedding, threshold=SIMILARITY_THRESHOLD)

        # --- Prepare text and drawing ---
        result_text = f"Similarity: {similarity_score:.2f}"
        match_text = "MATCH!" if is_same_person else "NO MATCH"
        box_color = (0, 255, 0) if is_same_person else (0, 0, 255) # Green for match, Red for no match

        # Draw bounding box
        cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), box_color, 2)

        # Put text
        # Place text slightly above the bounding box
        text_y_pos = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[1] + 25 # Ensure text is visible
        cv2.putText(display_frame, result_text, (bbox[0], text_y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)
        cv2.putText(display_frame, match_text, (bbox[0], text_y_pos + 20), # Below similarity
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

        print(f"{result_text} - {match_text}")


    except ValueError as ve: # Specifically for "No faces detected" or "Input image data is None"
        cv2.putText(display_frame, str(ve), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2) # Orange color for warnings
        print(f"Processing Info: {str(ve)}")
    except Exception as e:
        error_msg = f"Error: {str(e)[:50]}" # Truncate long errors for display
        cv2.putText(display_frame, error_msg, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) # Red for errors
        print(f"An unexpected error occurred: {str(e)}")

    # Display the resulting frame
    cv2.imshow('Webcam Face Comparison', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Application ended.")
