import cv2
import insightface
import numpy as np
import math
from insightface.app import FaceAnalysis
from videofeedtest import create_img_mask

# Initialize face analysis model
app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider'])  # Use 'CUDAExecutionProvider' for GPU
app.prepare(ctx_id=0)  # ctx_id=-1 for CPU, 0 for GPU

def get_img_embedding(img):
    faces = app.get(img)
    
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
    
    return faces[0].embedding

def get_face_embedding(image_path):
    """Extract face embedding from an image"""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    faces = app.get(img)
    
    if len(faces) < 1:
        raise ValueError("No faces detected in the image")
    if len(faces) > 1:
        print("Warning: Multiple faces detected. Using first detected face")
    
    return faces[0].embedding

def compare_faces(emb1, emb2, threshold=0.65): # Adjust this threshold according to your usecase.
    """Compare two embeddings using cosine similarity"""
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return similarity, similarity > threshold

image2_path = "Faces/test.jpg"
emb2 = get_face_embedding(image2_path)
print(f"total dimensions for vector embedding is: {len(emb2)}")
cap = cv2.VideoCapture(0) # Setup the OpenCV capture device (webcam)
while True:
    
    try:
        ret, frame = cap.read()
       
        # Get embeddings
        faces = app.get(frame)
        if len(faces) >= 1:
            # Compare faces
            emb1 = faces[0].embedding
            similarity_score, is_same_person = compare_faces(emb1, emb2)
            print(f"Similarity Score: {similarity_score:.4f}")
            print(f"Same person? {'YES' if is_same_person else 'NO'}")
            
            create_img_mask(frame, image2_path)
            box = faces[0].bbox.astype(int)
            if is_same_person:
                color = (0, 255, 0)
            else: 
                color = (0, 0, 255)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
        else:
            None

        cv2.imshow('WebCam', frame)
        if cv2.waitKey(1) == ord('q'):
            break
 
    except Exception as e:
        print(e)
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
