# importing the libraries
import cv2
import numpy as np
import math

# Setup camera

def create_img_mask(frame, img_str: str):
    # Load the face image (source face to overlay)
    srcface = cv2.imread(img_str)
    if srcface is None:
        raise ValueError(f"Could not load image from path: {img_str}")
    
    # Get dimensions of the source image
    dimensions = srcface.shape
    if len(dimensions) == 3:  # Color image
        height, width, channels = dimensions
    else:  # Grayscale image
        height, width = dimensions

    # Resize the source face image to fit within a smaller size
    size2 = 100
    size1 = math.floor(size2 * width / height)
    srcface_resized = cv2.resize(srcface, (size1, size2))

    # Create a mask from the source face image (convert to grayscale)
    img2gray = cv2.cvtColor(srcface_resized, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)

    # Define the region of interest (ROI) where we want to overlay the image on the frame
    roi = frame[-size2-10:-10, -size1-10:-10]

    # Apply the mask: only keep the non-zero parts of the face image
    roi[np.where(mask != 0)] = srcface_resized[np.where(mask != 0)]
    
    return None

if __name__ == "__main__" :
    cap = cv2.VideoCapture(0)
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        #create_img_mask(frame, 'Faces/test.jpg')
        cv2.imshow('WebCam', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()