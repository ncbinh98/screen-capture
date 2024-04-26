import cv2
import numpy as np
import mss
import mss.tools
import time
from ultralytics import YOLO
model = YOLO('./last.pt')  # load a custom model
# model.export(format='openvino') 
# ov_model = YOLO('last_openvino_model/')
# model = YOLO('last_openvino_model/', task='classify')
def capture_screen():
    with mss.mss() as sct:
        # Get the monitor's width and height
        monitor = sct.monitors[1]  # Change index if you have multiple monitors
        monitor_width = monitor["width"]
        monitor_height = monitor["height"]

        # Define the capture region
        capture_region = {
            "left": int(monitor_width * 0.5),
            "top":  int(monitor_height * 0.5),
            "width": int(monitor_width * 0.5),  # Reduce width by 50%
            "height": int(monitor_height * 0.5),  # Reduce height by 50%
        }
        
        prev_time = 0

        while True:
            # Capture the screen
            screenshot = sct.grab(capture_region)

            # Convert the screenshot to an OpenCV image
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
            # Calculate FPS
            current_time = time.time()
            fps = 1.0 / (current_time - prev_time)
            prev_time = current_time

            # Display FPS
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # AI GO
            results = model(frame)  # predict on an image
            names_dict = results[0].names
            probs = results[0].probs.data.tolist()
           
            
            cv2.putText(frame, f"predict: {names_dict[np.argmax(probs)]}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
       
            # Show the captured frame (optional)
            cv2.imshow('SCREEN', frame)

            # Exit the loop when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_screen()
