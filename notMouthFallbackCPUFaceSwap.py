import cv2
import os
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.model_zoo.inswapper import INSwapper

def get_available_providers():
    """Check available providers and return the best available option."""
    available_providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider"]
    elif "DmlExecutionProvider" in available_providers:  # DirectML for Windows
        return ["DmlExecutionProvider"]
    else:
        print("Warning: Running on CPU. This might be slower.")
        return ["CPUExecutionProvider"]

def initialize_face_analysis(providers):
    """Initialize FaceAnalysis with appropriate provider and ctx_id."""
    app = FaceAnalysis(
        name='buffalo_l',
        providers=providers,
        allowed_modules=["detection", "recognition"]
    )
    # Use ctx_id=-1 for CPU, 0 for GPU
    ctx_id = 0 if providers[0] == "CUDAExecutionProvider" else -1
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app

def create_mouth_mask(face_landmarks, frame_shape, dilation_factor=1.1):
    """Create a mask for the mouth region based on facial landmarks."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    
    # Extract mouth landmarks (last 20 points in InsightFace landmarks are for mouth)
    mouth_landmarks = face_landmarks[-20:].astype(np.int32)
    
    # Calculate center and size of mouth region
    center = np.mean(mouth_landmarks, axis=0)
    size = np.max(mouth_landmarks, axis=0) - np.min(mouth_landmarks, axis=0)
    
    # Scale the size by dilation_factor
    size = size * dilation_factor
    
    # Calculate box coordinates
    x1 = int(max(0, center[0] - size[0]/2))
    y1 = int(max(0, center[1] - size[1]/2))
    x2 = int(min(frame_shape[1], center[0] + size[0]/2))
    y2 = int(min(frame_shape[0], center[1] + size[1]/2))
    
    # Draw filled rectangle for mouth region
    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    
    # Blur the mask edges for smoother blending
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    
    return mask

def blend_with_original_mouth(original_frame, swapped_frame, face):
    """Blend the original mouth back into the swapped face."""
    try:
        # Create mouth mask
        mouth_mask = create_mouth_mask(face.landmark_2d_106, original_frame.shape)
        
        # Normalize mask to range 0-1
        mouth_mask_normalized = mouth_mask.astype(float) / 255
        
        # Expand dimensions to match image channels
        mouth_mask_normalized = np.expand_dims(mouth_mask_normalized, axis=-1)
        
        # Blend images based on mask
        result = swapped_frame.copy()
        result = (result * (1 - mouth_mask_normalized) + 
                 original_frame * mouth_mask_normalized).astype(np.uint8)
        
        return result
    except Exception as e:
        print(f"Error in blending mouth: {e}")
        return swapped_frame

def main():
    try:
        #############################################
        # 1) Configure paths and parameters
        #############################################
        source_image_path = "./models/hughJackman.jpg"
        inswapper_model_path = os.path.join("models", "inswapper_128.onnx")
        camera_index = 0
        capture_width = 640
        capture_height = 480

        #############################################
        # 2) Determine available providers and initialize
        #############################################
        providers = get_available_providers()
        
        # Initialize FaceAnalysis
        try:
            app = initialize_face_analysis(providers)
        except Exception as e:
            print(f"Error initializing FaceAnalysis: {e}")
            return

        #############################################
        # 3) Initialize the INSwapper
        #############################################
        try:
            session = ort.InferenceSession(inswapper_model_path, providers=providers)
            inswapper = INSwapper(model_file=inswapper_model_path, session=session)
        except Exception as e:
            print(f"Error initializing INSwapper: {e}")
            return

        #############################################
        # 4) Load and analyze source image
        #############################################
        source_bgr = cv2.imread(source_image_path)
        if source_bgr is None:
            raise FileNotFoundError(f"Cannot read source image: {source_image_path}")

        source_faces = app.get(source_bgr)
        if len(source_faces) == 0:
            raise ValueError("No face found in the source image.")
        source_face = source_faces[0]

        #############################################
        # 5) Open the webcam
        #############################################
        cap = cv2.VideoCapture(camera_index)
        
        # Try DirectShow backend on Windows
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, capture_height)

        print("Press 'q' to quit the video preview.")

        #############################################
        # 6) Main loop: read frames and swap faces
        #############################################
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                print("Failed to read frame from camera.")
                break

            try:
                target_faces = app.get(frame_bgr)

                if len(target_faces) > 0:
                    target_face = target_faces[0]
                    
                    # Perform the face swap
                    swapped_bgr = inswapper.get(
                        img=frame_bgr,
                        target_face=target_face,
                        source_face=source_face,
                        paste_back=True
                    )
                    
                    # Blend original mouth back
                    final_result = blend_with_original_mouth(frame_bgr, swapped_bgr, target_face)
                    
                    cv2.imshow("Face Swap (Webcam)", final_result)
                else:
                    cv2.imshow("Face Swap (Webcam)", frame_bgr)

            except Exception as e:
                print(f"Error processing frame: {e}")
                cv2.imshow("Face Swap (Webcam)", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        #############################################
        # 7) Cleanup
        #############################################
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()