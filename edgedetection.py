import cv2
import numpy as np

def show_step(title, image):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# -----------------------
# Step 1: Load + preprocess for parchment detection
# -----------------------
img = cv2.imread('/Users/scottgarciajr/Desktop/Programming/Dead Sea Scrolls/Greek_Man_Samp.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)

# Otsu threshold
_, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# ✅ Invert so fragments = white, background = black
otsu_inv = cv2.bitwise_not(otsu)

# Morphology cleanup
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
mask = cv2.morphologyEx(otsu_inv, cv2.MORPH_CLOSE, kernel, iterations=2)

cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Step 2: Find fragment contours
# -----------------------
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter out tiny junk
contours = [c for c in contours if cv2.contourArea(c) > 2000]

print(f"Extracted {len(contours)} parchment fragments")

# Debug view: draw detected contours
debug_view = img.copy()
cv2.drawContours(debug_view, contours, -1, (0,255,0), 2)
cv2.imshow("Detected Fragments", debug_view)
cv2.waitKey(0)
cv2.destroyAllWindows()


# -----------------------
# Step 3: User reference shape (draw)
# -----------------------
drawing = False
points = []

def draw_shape(event, x, y, flags, param):
    global drawing, points
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        points = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        points.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        points.append((x, y))

cv2.namedWindow("Draw Shape")
cv2.setMouseCallback("Draw Shape", draw_shape)

canvas = np.ones(img.shape, dtype=np.uint8) * 255
while True:
    temp = canvas.copy()
    if len(points) > 1:
        cv2.polylines(temp, [np.array(points)], False, (0,0,0), 2)
    cv2.imshow("Draw Shape", temp)
    key = cv2.waitKey(1) & 0xFF
    if key == 13:  # Enter
        break

# Make contour from drawn shape
drawn_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
if len(points) > 2:
    cv2.fillPoly(drawn_mask, [np.array(points)], 255)

user_contours, _ = cv2.findContours(drawn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if not user_contours:
    print("No contour drawn.")
    exit()
user_contour = max(user_contours, key=cv2.contourArea)

# -----------------------
# Step 4: Match with fragments
# -----------------------
best_score = float("inf")
best_contour = None

for c in contours:
    score = cv2.matchShapes(user_contour, c, cv2.CONTOURS_MATCH_I1, 0)
    if score < best_score:
        best_score = score
        best_contour = c

print(f"Best match score: {best_score:.4f}")

# -----------------------
# Step 5: Display results
# -----------------------

# All fragments (green outlines)
all_fragments_view = img.copy()
cv2.drawContours(all_fragments_view, contours, -1, (0,255,0), 2)

# Best match (red highlight)
best_match_view = img.copy()
if best_contour is not None:
    cv2.drawContours(best_match_view, [best_contour], -1, (0,0,255), 3)
    x, y, w, h = cv2.boundingRect(best_contour)
    cv2.putText(best_match_view, f"Best Match (score={best_score:.3f})", 
                (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

# -----------------------
# Step 5b: Show all fragments in a grid with edges
# -----------------------
fragment_images = []
for i, c in enumerate(contours):
    x, y, w, h = cv2.boundingRect(c)
    frag = img[y:y+h, x:x+w].copy()

    mask_frag = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask_frag, [c - [x, y]], -1, 255, -1)
    frag_masked = cv2.bitwise_and(frag, frag, mask=mask_frag)

    edges = cv2.Canny(mask_frag, 50, 150)
    frag_edges = frag_masked.copy()
    frag_edges[edges > 0] = (0, 0, 255)
    fragment_images.append(frag_edges)

if fragment_images:
    row_size = 4
    max_h = max(f.shape[0] for f in fragment_images)
    max_w = max(f.shape[1] for f in fragment_images)

    # pad all fragments to same size
    padded = []
    for f in fragment_images:
        h, w = f.shape[:2]
        pad_top = 0
        pad_bottom = max_h - h
        pad_left = 0
        pad_right = max_w - w
        f_padded = cv2.copyMakeBorder(f, pad_top, pad_bottom, pad_left, pad_right,
                                      cv2.BORDER_CONSTANT, value=(255,255,255))
        padded.append(f_padded)

    rows = []
    for i in range(0, len(padded), row_size):
        row = padded[i:i+row_size]
        rows.append(np.hstack(row))
    grid = np.vstack(rows)

    cv2.imshow("Fragments Grid", grid)

# Show combined results
cv2.imshow("All Fragments (Green)", all_fragments_view)
cv2.imshow("Best Match (Red)", best_match_view)
cv2.waitKey(0)
cv2.destroyAllWindows()
