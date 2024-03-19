import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)  # '0' is usually the default ID for the built-in camera

errorMargin = 20  # Margin for center comparison
radiusMargin = 10  # Margin for radius difference

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Calculate the center of the frame
    height, width = frame.shape[:2]
    center_coordinates = (width // 2, height // 2)
    center_width = width // 2
    center_height = height // 2
    
    # Convert from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for red color and create mask
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)

    # Define range for white color and create mask
    lower_white = np.array([0, 0, 168])
    upper_white = np.array([172, 111, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Apply Gaussian blur to reduce noise and improve circle detection for both masks
    blurred_red = cv2.GaussianBlur(mask_red, (9, 9), 2)
    blurred_white = cv2.GaussianBlur(mask_white, (9, 9), 2)

    # Detect circles in red mask
    circles_red = cv2.HoughCircles(blurred_red, cv2.HOUGH_GRADIENT, dp=1, minDist=250,
                                   param1=45, param2=55, minRadius=0, maxRadius=0)

    # Detect circles in white mask
    circles_white = cv2.HoughCircles(blurred_white, cv2.HOUGH_GRADIENT, dp=1, minDist=250,
                                     param1=45, param2=55, minRadius=0, maxRadius=0)

    if circles_red is not None and circles_white is not None:
        circles_red = np.uint16(np.around(circles_red))
        circles_white = np.uint16(np.around(circles_white))

        # Find matching centers and ensure the white circle is significantly smaller than the red circle
        for red_circle in circles_red[0, :]:
            for white_circle in circles_white[0, :]:
                distance = np.sqrt((red_circle[0] - white_circle[0])**2 + (red_circle[1] - white_circle[1])**2)
                radius_difference = red_circle[2] - white_circle[2]
                
                if distance < errorMargin and radius_difference > radiusMargin:
                    # Found a white circle within a red circle with significant size difference
                    cv2.circle(frame, (red_circle[0], red_circle[1]), red_circle[2], (0, 0, 255), 2)  # Draw red circle
                    cv2.circle(frame, (white_circle[0], white_circle[1]), white_circle[2], (0, 255, 0), 2)  # Draw white circle
                    cv2.circle(frame, (red_circle[0], red_circle[1]), 2, (255, 255, 255), 3)  # Mark center
                    
                    
                    if(center_width - errorMargin < red_circle[0] < center_width + errorMargin) and (red_circle[1] < center_height + errorMargin):
                        string1 = "TRUE"
                    elif(red_circle[0] > center_width + errorMargin):
                        string1 = "LEFT"
                    else:
                        string1 = "RIGHT"

                    if(string1 == "TRUE") and (red_circle[1] > center_height - errorMargin):
                        string2 = "TRUE"
                    else:
                        string2 = "FALSE"

                    cv2.putText(frame, string1, (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    cv2.putText(frame, string2, (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    cv2.imshow('Circle Detection', frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()