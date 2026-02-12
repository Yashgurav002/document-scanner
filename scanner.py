import cv2
import numpy as np
import imutils
import os


def order_points(pts):
    """
    Reorder 4 points in the order:
    top-left, top-right, bottom-right, bottom-left
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # top-left
    rect[2] = pts[np.argmax(s)]   # bottom-right

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


# -----------------------------
# 1. Load Image
# -----------------------------
image_path = "input/document4.jpg"

if not os.path.exists(image_path):
    raise FileNotFoundError("❌ input/document.jpg not found")

image = cv2.imread(image_path)

if image is None:
    raise Exception("❌ OpenCV could not read the image")

orig = image.copy()

# -----------------------------
# 2. Resize with Ratio Tracking
# -----------------------------
ratio = image.shape[0] / 500.0
image = imutils.resize(image, height=500)

# -----------------------------
# 3. Preprocessing
# -----------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# -----------------------------
# 4. Edge Detection
# -----------------------------
edged = cv2.Canny(gray, 50, 150)

# -----------------------------
# 5. Find Contours
# -----------------------------
contours = cv2.findContours(
    edged.copy(),
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)

# -----------------------------
# 6. Find Document Contour
# -----------------------------
screenCnt = None

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        screenCnt = approx
        break

if screenCnt is None:
    raise Exception("❌ Could not detect document contour")

# -----------------------------
# 7. Perspective Transform
# -----------------------------
pts = screenCnt.reshape(4, 2) * ratio
rect = order_points(pts)
(tl, tr, br, bl) = rect

widthA = np.linalg.norm(br - bl)
widthB = np.linalg.norm(tr - tl)
maxWidth = int(max(widthA, widthB))

heightA = np.linalg.norm(tr - br)
heightB = np.linalg.norm(tl - bl)
maxHeight = int(max(heightA, heightB))

dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]
], dtype="float32")

M = cv2.getPerspectiveTransform(rect, dst)
warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))

# -----------------------------
# 8. Scan Outputs (MULTIPLE MODES)
# -----------------------------
os.makedirs("output", exist_ok=True)

# 8A. Clean Grayscale (BEST FOR OCR)
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
cv2.imwrite("output/scanned_gray.jpg", warped_gray)

# 8B. OTSU Threshold (STABLE)
_, scanned_otsu = cv2.threshold(
    warped_gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)
cv2.imwrite("output/scanned_otsu.jpg", scanned_otsu)

# 8C. Adaptive Threshold (OPTIONAL)
scanned_adaptive = cv2.adaptiveThreshold(
    warped_gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    21,
    5
)
cv2.imwrite("output/scanned_adaptive.jpg", scanned_adaptive)

print("✅ Document scanning complete!")
print("📁 Outputs saved in /output folder")
