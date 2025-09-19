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

# Invert so fragments = white, background = black
otsu_inv = cv2.bitwise_not(otsu)

# Morphology cleanup (gentler for small fragments)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
mask = cv2.morphologyEx(otsu_inv, cv2.MORPH_OPEN, kernel, iterations=1)  # remove specks
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)      # close small holes

cv2.imshow("Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# -----------------------
# Step 2: Find fragment contours
# -----------------------
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Lower the area filter to keep small fragments
contours = [c for c in contours if cv2.contourArea(c) > 200]

print(f"Extracted {len(contours)} parchment fragments (including small ones)")

# Debug view
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
# Step 5b: Show all fragments individually in a grid
# -----------------------
fragment_images = []
for i, c in enumerate(contours):
    x, y, w, h = cv2.boundingRect(c)
    frag = img[y:y+h, x:x+w].copy()

    # Optional: mask out background for cleaner display
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.drawContours(mask, [c - [x, y]], -1, 255, -1)
    frag_masked = cv2.bitwise_and(frag, frag, mask=mask)

    fragment_images.append(frag_masked)

# Make a grid for visualization
if fragment_images:
    rows = []
    row_size = 4  # number of fragments per row
    for i in range(0, len(fragment_images), row_size):
        row = fragment_images[i:i+row_size]
        # Resize to same height
        max_h = max(f.shape[0] for f in row)
        row_resized = [cv2.copyMakeBorder(
            f, 0, max_h - f.shape[0], 0, 0,
            cv2.BORDER_CONSTANT, value=(255,255,255)) 
            for f in row]
        row_img = np.hstack(row_resized)
        rows.append(row_img)

    # --- Fix: pad rows to same width ---
    max_w = max(r.shape[1] for r in rows)
    rows_padded = [cv2.copyMakeBorder(
        r, 0, 0, 0, max_w - r.shape[1],
        cv2.BORDER_CONSTANT, value=(255,255,255))
        for r in rows]

    grid = np.vstack(rows_padded)
    cv2.imshow("Fragments Grid", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("No fragments extracted to display.")


# Show combined results
cv2.imshow("All Fragments (Green)", all_fragments_view)
cv2.imshow("Best Match (Red)", best_match_view)
cv2.waitKey(0)
cv2.destroyAllWindows()
