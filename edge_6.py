
######### This version for both circles and red ###############

import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)  # '0' is usually the default ID for the built-in camera

errorMargin = 20


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
    
    # Draw a blue dot (circle) at the center of the frame
    # Parameters: image, center_coordinates, radius, color(BGR), thickness
    cv2.circle(frame, center_coordinates, 5, (255, 0, 0), -1)  # Blue dot

    # Display the resulting frame
    cv2.imshow('frame', frame)

    # Convert from BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

 # Define range for red color and create masks
    ## the numbers here stand for (hue, saturation, value)
    lower_red1 = np.array([0, 50, 50]) 
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([160, 50, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    # Combine the masks
    mask = mask1 + mask2

    #this is applying the edge detection
    edges = cv2.Canny(mask, 100, 200)

 # Convert to grayscale
   # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    value_channel = mask

    # Apply Gaussian blur to reduce noise and improve circle detection
    value_channel_blurred = cv2.GaussianBlur(value_channel, (9, 9), 2)



    # Detect circles in the image
    #the higher param2 is the less sensitive it is to detecting circles 
    circles = cv2.HoughCircles(value_channel_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=250,
                               param1=45, param2=55, minRadius=0, maxRadius=0)

    #Line facing upwards
    #cv2.line(frame,center_coordinates,(center_width,0),(0, 255, 0), 3)
    # Ensure at least some circles were found
    if circles is not None:
        # Convert the circle parameters (x, y, radius) to integers
        circles = np.uint16(np.around(circles))

        # Loop over the (x, y) coordinates and radius of the circles
        for i in circles[0, :]:
            # Draw the circle in the output frame
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw the center of the circle
            cv2.circle(frame, (i[0], i[1]), 2, (0, 0, 255), 3)
            #displays the coordinates of the center of the circle on camera
            cv2.putText(frame, f"({i[0]}, {i[1]})", (i[0]+5, i[1]+5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            cv2.line(frame,center_coordinates,(i[0],i[1]),(0, 255, 0), 3)
            if((i[0]>(center_width-errorMargin))&(i[0]<(center_width+errorMargin))&(i[1]<(center_height+errorMargin))):
               string1 = "TRUE"
            elif(i[0]>(center_width-errorMargin)):
                string1 = "LEFT"
            else:
                string1 = "RIGHT"
                

            if((string1 == "TRUE")&(i[1]>(center_height-errorMargin))):
                string2 = "TRUE"
            else:
                string2 = "FALSE"

            

            cv2.putText(frame,string1,(5,30),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame,string2,(5,80),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            # displays the center of the frame minus the coordinates of the center of the 
            # circle to output our distance from the center 
           # distance_from_center = center_coordinates - ({i[0]}, {i[1]})
           # cv2.putText(distance_from_center, (120, 400),2 (0, 255, 255), 2)
           # cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255, 2)
    
           
           
    
    # Display the resulting frame
    cv2.imshow('Circle Detection', frame)
    cv2.imshow('red circle',mask)
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
