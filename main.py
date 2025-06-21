import cv2
import numpy as np
from utils import get_parking_boxes, parking_space
import warnings

warnings.filterwarnings("ignore")


def calc_diff(im1: np.array, im2: np.array)->float:
    """
    Sees if there is a large change in pixel/intensity in the two frames, indicating change in status
    :param im1:
    :param im2:
    :return:
    """
    return np.abs(np.mean(im1) - np.mean(im2))

mask = cv2.imread('./data/mask.png', cv2.IMREAD_GRAYSCALE) # can also use cv2.imread('path', 0)
vid = cv2.VideoCapture('./data/parking_vid.mp4')
last_frame = None


cc = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
#returns image where all the pixels which are connected are labeled together (in this case the white parts of parkin
# lot). 4 means direction of connectivity (up, down, left, right). 8 would include diagonals

slots = get_parking_boxes(cc)

ret = True
interval = 30
frame_num = 0
all_status = [None for slot in slots]
diff = [None for s in slots]

while ret:
    ret, frame = vid.read()
    # ret -> True if the frame was read successfully


    if frame_num%interval == 0 and last_frame is not None:
        for i, slot in enumerate(slots):
            x1, y1, w, h, _ = slot
            x2, y2 = x1 + w, y1 + h
            space = frame[y1:y2, x1:x2, :]
            last_space = last_frame[y1:y2, x1:x2, :]

            diff[i] = calc_diff(space, last_space)


    if frame_num % interval == 0:
        if last_frame is None:
            arr_opt = range(len(slots))
        else:
            arr_opt = [n for n in np.argsort(diff) if diff[n] / np.amax(diff) > 0.4]

        last_frame = frame.copy()

        #creating rectangle for each spot
        for i in arr_opt:
            x1, y1, w, h, _ = slots[i]
            x2, y2 = x1+w, y1+h
            space = frame[y1:y2, x1:x2, :]
            status = parking_space(space)
            all_status[i] = status

    for i, slot in enumerate(slots):
        x1, y1, w, h, center = slots[i]
        x2, y2 = x1+w, y1+h
        cx, cy = center
        if all_status[i]:
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            frame = cv2.circle(frame, (int(cx), int(cy)), radius=3, color=(128, 0, 128), thickness=-1)

        else:
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            frame = cv2.circle(frame, (int(cx), int(cy)), radius=3, color=(128, 0, 128), thickness=-1)

    frame_num += 1

    if not ret:
        print("End of video or failed to read frame.")
        break

    cv2.putText(frame, f'Available Spots: {sum(all_status)}/{len(all_status)}', (100, 60),
                cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
    cv2.namedWindow('ParkingFootage', cv2.WINDOW_NORMAL)
    cv2.imshow('Parking Footage', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'): # basically closes the frame when we hit q
        break
    # waits 25ms for a keypress (q) -> closes the window

vid.release()
cv2.destroyAllWindows()