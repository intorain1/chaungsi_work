import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread(r"C:\Users\57704\Desktop\tri_pre\data\lslm\images\train\C_Users_57704_Desktop_tri_pre_data_lslm_images_train_383.jpg") #TODO change to your path

img2 = img.copy()
img3 = img.copy()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny = cv2.Canny(gray, 120, 255)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
dilated = cv2.dilate(canny, kernel)

contours, hierarchy = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for contour in contours:

    rrect = cv2.minAreaRect(contour)
    points = cv2.boxPoints(rrect)  
    points = np.int0(points)  
    center = rrect[0] 
    width, height = max(rrect[1]), min(rrect[1])
    area = width * height
    print(f"min 长: {height}, 宽: {width}")

    area1 = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    circularity = 4 * np.pi * area1 / (perimeter ** 2) if perimeter > 0 else 0

    print(circularity)
    if (width < 1.2 * height) and area > 5000 and circularity < 0.9 :
        print('hello')
        cv2.polylines(img2, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.circle(img2, (int(center[0]), int(center[1])), 2, (255, 0, 0), 2)
        cv2.putText(img2, "nut", (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if (width > 1.2 * height) and area > 20000:
        cv2.polylines(img2, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.circle(img2, (int(center[0]), int(center[1])), 2, (255, 0, 0), 2)
        cv2.putText(img2, "bolt", (int(center[0]), int(center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (11, 11), 2)
edges = cv2.Canny(blurred, 100, 150)

circles = cv2.HoughCircles(
    edges,
    cv2.HOUGH_GRADIENT,
    dp=1,  
    minDist=400,  
    param1=70,  
    param2=20,  
    minRadius=10, 
    maxRadius=450  
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        cv2.circle(img2, (i[0], i[1]), i[2], (0, 255, 0), 2)
        cv2.putText(img2, "coin", ((i[0], i[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

img_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

plt.imshow(img_rgb)
plt.axis('off')  
plt.title('Detected Objects')
plt.show()
