import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, QFileDialog,
                             QVBoxLayout, QWidget, QHBoxLayout, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

# -----------------------
# Helper: convert OpenCV image to QPixmap
# -----------------------
def cv2_to_qpixmap(cv_img):
    rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb_image.shape
    bytes_per_line = ch * w
    qimage = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimage)

# -----------------------
# Main GUI
# -----------------------
class FragmentMatcher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dead Sea Scroll Fragment Matcher")
        self.setGeometry(100, 100, 1400, 800)

        self.img = None
        self.fragments = []
        self.best_contour = None
        self.user_contour = None

        # Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        # Left: Image display
        self.image_label = QLabel("Load an image to start")
        self.image_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.image_label, 3)

        # Right: Control buttons
        control_layout = QVBoxLayout()
        layout.addLayout(control_layout, 1)

        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)
        control_layout.addWidget(self.load_button)

        self.detect_button = QPushButton("Detect Fragments")
        self.detect_button.clicked.connect(self.detect_fragments)
        self.detect_button.setEnabled(False)
        control_layout.addWidget(self.detect_button)

        self.draw_button = QPushButton("Draw Reference Shape")
        self.draw_button.clicked.connect(self.draw_reference_shape)
        self.draw_button.setEnabled(False)
        control_layout.addWidget(self.draw_button)

        self.match_button = QPushButton("Match Shape")
        self.match_button.clicked.connect(self.match_shape)
        self.match_button.setEnabled(False)
        control_layout.addWidget(self.match_button)

        self.show_grid_button = QPushButton("Show Fragments Grid")
        self.show_grid_button.clicked.connect(self.show_fragments_grid)
        self.show_grid_button.setEnabled(False)
        control_layout.addWidget(self.show_grid_button)

    # -----------------------
    # Load Image
    # -----------------------
    def load_image(self):
        fname, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg)")
        if fname:
            self.img = cv2.imread(fname)
            pix = cv2_to_qpixmap(self.img)
            self.image_label.setPixmap(pix.scaled(800, 800, Qt.KeepAspectRatio))
            self.detect_button.setEnabled(True)

    # -----------------------
    # Detect fragments
    # -----------------------
    def detect_fragments(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_inv = cv2.bitwise_not(otsu)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask = cv2.morphologyEx(otsu_inv, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.fragments = [c for c in contours if cv2.contourArea(c) > 200]

        # Draw fragments
        display = self.img.copy()
        cv2.drawContours(display, self.fragments, -1, (0,255,0), 2)
        pix = cv2_to_qpixmap(display)
        self.image_label.setPixmap(pix.scaled(800, 800, Qt.KeepAspectRatio))

        QMessageBox.information(self, "Fragments Detected", f"Detected {len(self.fragments)} fragments")
        self.draw_button.setEnabled(True)

    # -----------------------
    # Draw reference shape
    # -----------------------
    def draw_reference_shape(self):
        if self.img is None:
            return
        self.drawing = False
        self.points = []

        canvas = self.img.copy()
        window_name = "Draw Shape - Press Enter to confirm"
        cv2.namedWindow(window_name)

        def draw_shape(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.drawing = True
                self.points = [(x, y)]
            elif event == cv2.EVENT_MOUSEMOVE and self.drawing:
                self.points.append((x, y))
            elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                self.points.append((x, y))

        cv2.setMouseCallback(window_name, draw_shape)

        while True:
            temp = canvas.copy()
            if len(self.points) > 1:
                cv2.polylines(temp, [np.array(self.points)], False, (0,0,0), 2)
            cv2.imshow(window_name, temp)
            key = cv2.waitKey(1) & 0xFF
            if key == 13:  # Enter
                break

        cv2.destroyWindow(window_name)

        drawn_mask = np.zeros(canvas.shape[:2], dtype=np.uint8)
        if len(self.points) > 2:
            cv2.fillPoly(drawn_mask, [np.array(self.points)], 255)
        user_contours, _ = cv2.findContours(drawn_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not user_contours:
            QMessageBox.warning(self, "No Shape", "No contour drawn")
            return
        self.user_contour = max(user_contours, key=cv2.contourArea)
        self.match_button.setEnabled(True)

    # -----------------------
    # Match shape
    # -----------------------
    def match_shape(self):
        if self.user_contour is None or not self.fragments:
            return
        best_score = float("inf")
        best_contour = None
        for c in self.fragments:
            score = cv2.matchShapes(self.user_contour, c, cv2.CONTOURS_MATCH_I1, 0)
            if score < best_score:
                best_score = score
                best_contour = c
        self.best_contour = best_contour

        # Display results
        display = self.img.copy()
        cv2.drawContours(display, self.fragments, -1, (0,255,0), 2)
        if self.best_contour is not None:
            cv2.drawContours(display, [self.best_contour], -1, (0,0,255), 3)
            x, y, w, h = cv2.boundingRect(self.best_contour)
            cv2.putText(display, f"Best Match (score={best_score:.3f})", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        pix = cv2_to_qpixmap(display)
        self.image_label.setPixmap(pix.scaled(800, 800, Qt.KeepAspectRatio))
        QMessageBox.information(self, "Match Result", f"Best match score: {best_score:.4f}")
        self.show_grid_button.setEnabled(True)

    # -----------------------
    # Show fragments grid
    # -----------------------
    def show_fragments_grid(self):
        if not self.fragments:
            return

        fragment_images = []
        for c in self.fragments:
            x, y, w, h = cv2.boundingRect(c)
            frag = self.img[y:y+h, x:x+w].copy()
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [c - [x, y]], -1, 255, -1)
            frag_masked = cv2.bitwise_and(frag, frag, mask=mask)
            fragment_images.append(frag_masked)

        # Make grid
        row_size = 4
        max_h = max(f.shape[0] for f in fragment_images)
        max_w = max(f.shape[1] for f in fragment_images)
        padded = []
        for f in fragment_images:
            h, w = f.shape[:2]
            f_pad = cv2.copyMakeBorder(f, 0, max_h - h, 0, max_w - w, cv2.BORDER_CONSTANT, value=(255,255,255))
            padded.append(f_pad)

        rows = []
        for i in range(0, len(padded), row_size):
            row = padded[i:i+row_size]
            row_img = np.hstack(row)
            rows.append(row_img)
        max_row_w = max(r.shape[1] for r in rows)
        rows_padded = [cv2.copyMakeBorder(r, 0, 0, 0, max_row_w - r.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255)) for r in rows]
        grid = np.vstack(rows_padded)

        # Highlight best match in grid
        if self.best_contour is not None:
            x_best, y_best, w_best, h_best = cv2.boundingRect(self.best_contour)
            for i, c in enumerate(self.fragments):
                x, y, w, h = cv2.boundingRect(c)
                if x == x_best and y == y_best:
                    # Draw a red border around this fragment in the grid
                    fragment_images[i] = cv2.copyMakeBorder(fragment_images[i], 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0,0,255))

        # Display grid in a new window
        cv2.imshow("Fragments Grid", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# -----------------------
# Run the GUI
# -----------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FragmentMatcher()
    window.show()
    sys.exit(app.exec_())
