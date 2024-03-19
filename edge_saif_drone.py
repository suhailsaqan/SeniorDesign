import cv2
import numpy as np
import time
from dronekit import connect, VehicleMode
from pymavlink import mavutil  # Needed for command message definitions

# Connect to the Vehicle
vehicle = connect('/dev/ttyAMA0', wait_ready=True, baud=57600)
vehicle.mode = VehicleMode("GUIDED")

# Initialize the camera
cap = cv2.VideoCapture(0)  # '0' for the default camera

errorMargin = 20  # Margin for error in circle center comparison
radiusMargin = 10  # Margin for radius difference in circle detection

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Calculate the center of the frame
    height, width = frame.shape[:2]
    center_width, center_height = width // 2, height // 2

    # Convert from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define range for red and white colors and create masks
    lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 50, 50]), np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
    
    lower_white, upper_white = np.array([0, 0, 168]), np.array([172, 111, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Apply Gaussian blur to reduce noise and improve circle detection
    blurred_red = cv2.GaussianBlur(mask_red, (9, 9), 2)
    blurred_white = cv2.GaussianBlur(mask_white, (9, 9), 2)

    # Detect circles in both masks
    circles_red = cv2.HoughCircles(blurred_red, cv2.HOUGH_GRADIENT, dp=1, minDist=250,
                                   param1=45, param2=55, minRadius=0, maxRadius=0)
    circles_white = cv2.HoughCircles(blurred_white, cv2.HOUGH_GRADIENT, dp=1, minDist=250,
                                     param1=45, param2=55, minRadius=0, maxRadius=0)

    if circles_red is not None and circles_white is not None:
        circles_red = np.uint16(np.around(circles_red))
        circles_white = np.uint16(np.around(circles_white))

        for red_circle in circles_red[0, :]:
            for white_circle in circles_white[0, :]:
                distance = np.sqrt((red_circle[0] - white_circle[0])**2 + (red_circle[1] - white_circle[1])**2)
                radius_difference = red_circle[2] - white_circle[2]
                
                if distance < errorMargin and radius_difference > radiusMargin:
                    cv2.circle(frame, (red_circle[0], red_circle[1]), red_circle[2], (0, 0, 255), 2)
                    cv2.circle(frame, (white_circle[0], white_circle[1]), white_circle[2], (0, 255, 0), 2)
                    cv2.circle(frame, (red_circle[0], red_circle[1]), 2, (255, 255, 255), 3)
                    
                    # Calculate distance from center
                    distanceX = red_circle[0] - center_width
                    distanceY = red_circle[1] - center_height

                    payloadAlt = 0.5  # altitude in meters. can change later
                    payloadLocation = vehicle.location.global_relative_frame
                    payloadLocation.alt = payloadAlt  # setting attribute

                    servoChannel = 9  # change to whatever input channel the servo motor is on
                    servoPWMValue = 2000  # Change to the desired PWM value to trigger the servo

                    if (abs(distanceX) > errorMargin):
                        #send message to pixhawk saying move the drone right or left
                        vehicle.simple_goto(LocationGlobalRelative(vehicle.location.global_relative_frame.lat, vehicle.location.global_relative_frame.lon + distanceX, vehicle.location.global_relative_frame.alt))
                    if (abs(distanceY) > errorMargin):
                        #send message to pixhawk saying move the drone backwards or forwards
                        vehicle.simple_goto(LocationGlobalRelative(vehicle.location.global_relative_frame.lat + distanceY, vehicle.location.global_relative_frame.lon, vehicle.location.global_relative_frame.alt))
                    if ((abs(distanceX) < errorMargin) & (abs(distanceY) < errorMargin) & (vehicle.location.global_relative_frame.alt > 6)):
                        #only move down 5 meters at a time. Only when the three conditions are met
                        vehicle.simple_goto(LocationGlobalRelative(vehicle.location.global_relative_frame.lat, vehicle.location.global_relative_frame.lon + distanceX, vehicle.location.global_relative_frame.alt - 5))
                    elif((abs(distanceX) < errorMargin) & (abs(distanceY) < errorMargin) & (vehicle.location.global_relative_frame.alt < 6)):
                        vehicle.simple_goto(payloadLocation)
                        while not vehicle.location.global_relative_frame.alt >= payloadAlt:
                            time.sleep(1) #while we wait for the drone to reach 0.5 meters, sleep. then check this condition again
                        vehicle.channels.overrides[servoChannel] = servoPWMValue
                        time.sleep(5)
                        vehicle.channels.overrides[servoChannel] = None

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
