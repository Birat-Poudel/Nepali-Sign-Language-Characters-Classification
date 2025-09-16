import os
import cv2

# Define directories
train_directory = "data/total"

# Create base directories if they do not exist
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists(train_directory):
    os.makedirs(train_directory)

# Define character lists
characters = [
    '0', '1', '2', '3', '4',
    '5', '6', '7', '8',]

# Create sub-folders for each character in train directories
for char in characters:
    os.makedirs(f"{train_directory}/{char}", exist_ok=True)

# Initialize camera
cap = cv2.VideoCapture(0)

min_value = 70
num_images_to_capture = 1000
image_count = 0
capture_images = False
capture_char = ''

# Initialize count dictionaries
train_chars_count = {char: len(os.listdir(f"{train_directory}/{char}")) for char in characters}

# Create a window
cv2.namedWindow("Parameters")

# Function to update parameters from trackbars
def nothing(x):
    pass

# Create trackbars for adjusting parameters
cv2.createTrackbar("Gaussian Kernel", "Parameters", 1, 20, nothing)
cv2.createTrackbar("Bilateral Filter d", "Parameters", 9, 20, nothing)
cv2.createTrackbar("Bilateral Filter sigmaColor", "Parameters", 75, 150, nothing)
cv2.createTrackbar("Bilateral Filter sigmaSpace", "Parameters", 75, 150, nothing)
cv2.createTrackbar("Adaptive Threshold Block Size", "Parameters", 11, 50, nothing)
cv2.createTrackbar("Adaptive Threshold C", "Parameters", 1, 20, nothing)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    cv2.rectangle(frame, (269, 9), (621, 355), (265, 0, 0), 1)
    cv2.imshow("Frame", frame)

    roi = frame[50:350, 270:570]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Get current positions of the trackbars
    ksize = cv2.getTrackbarPos("Gaussian Kernel", "Parameters")
    bilateral_d = cv2.getTrackbarPos("Bilateral Filter d", "Parameters")
    bilateral_sigmaColor = cv2.getTrackbarPos("Bilateral Filter sigmaColor", "Parameters")
    bilateral_sigmaSpace = cv2.getTrackbarPos("Bilateral Filter sigmaSpace", "Parameters")
    adaptive_blockSize = cv2.getTrackbarPos("Adaptive Threshold Block Size", "Parameters")
    adaptive_C = cv2.getTrackbarPos("Adaptive Threshold C", "Parameters")

    # Ensure kernel sizes are odd and greater than 1
    ksize = max(3, ksize | 1)
    adaptive_blockSize = max(3, adaptive_blockSize | 1)

    blur = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    blur = cv2.bilateralFilter(blur, bilateral_d, bilateral_sigmaColor, bilateral_sigmaSpace)
    th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, adaptive_blockSize, adaptive_C)
    _, roi = cv2.threshold(th3, min_value, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow("ROI", roi)

    interrupt = cv2.waitKey(100)  # Increase wait time to 100 milliseconds

    if interrupt & 0xFF == 27:  # ESC key
        break

    if not capture_images:
        for char in characters:
            if interrupt & 0xFF == ord(char.lower()):
                capture_images = True
                capture_char = char
                image_count = 0
                break
    else:
        if image_count % 5 == 0:  # Capture every 5th frame
            file_path = f"{train_directory}/{capture_char}/{train_chars_count[capture_char]}.jpg"
            train_chars_count[capture_char] += 1
            cv2.imwrite(file_path, roi)
        image_count += 1
        if image_count >= num_images_to_capture * 5:  # Adjust for frame skip
            capture_images = False
            break  # Exit the loop after capturing the desired number of images

cap.release()
cv2.destroyAllWindows()
