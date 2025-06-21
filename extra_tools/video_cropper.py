import cv2

# --- INPUT CONFIG ---
input_path = "../data/parking_vid.mp4"
output_path = "../data/parking_small.mp4"

# Crop rectangle (x, y, width, height)
crop_x = 0
crop_y = 0
crop_w = 474
crop_h = 1080

# --- OPEN VIDEO ---
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')


out = cv2.VideoWriter(output_path, fourcc, fps, (crop_w, crop_h))

ret = True
while ret:
    ret, frame = cap.read()

    # Crop the frame
    cropped = frame[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

    out.write(cropped)

# --- CLEANUP ---
cap.release()
out.release()
cv2.destroyAllWindows()

print("✅ Cropping complete. Saved to", output_path)
