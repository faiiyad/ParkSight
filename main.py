import cv2
import numpy as np
from utils import get_parking_boxes, parking_space
import warnings

warnings.filterwarnings("ignore")


# TODO: ENTER SECTION HERE
section = 'b'

mask_path = f'./data/section_{section}/mask.png'
road_path = f'./data/section_{section}/road.png'
vid_path = f'./data/section_{section}/vid.mp4'

mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)# can also use cv2.imread('path', 0)
road_mask = cv2.imread(road_path, 0)
vid = cv2.VideoCapture(vid_path)


w, h, fps = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vid.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(f'./output/output_{section}.mp4', fourcc, fps, (w,h))


cc = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
rcc = cv2.connectedComponentsWithStats(road_mask, 4, cv2.CV_32S)
#returns image where all the pixels which are connected are labeled together (in this case the white parts of parkin
# lot). 4 means direction of connectivity (up, down, left, right). 8 would include diagonals

slots = get_parking_boxes(cc)
roads = get_parking_boxes(rcc)

ret = True
interval = 30 # check for movement every 30frames (1 second)
frame_num = 0 # iterator
top_y = 80 # starting point
last_frame = None

diff = [None for s in slots] # difference between two frames (cropped to parking spot)
score = {} # initial score of each center

center_spots = {slot[-1] for slot in slots} # stores all the center locations for a faster lookup

center_status = {slot[-1]:1 for slot in slots} # stores if the given center is occupied or not

final_score = {slot[-1]:100000 for slot in slots} # stores the score for the pathfinding algorithm

center_status[(0, 0)] = True # edges (top and bottom) are assumed to be neighbours with (0,0) and
                             # get a boost (since it is easier to park in an edge)


def calc_diff(im1: np.array, im2: np.array)->float:
    """
    Sees if there is a large change in pixel/intensity in the two frames, indicating change in status
    :param im1:
    :param im2:
    :return:
    """
    return np.abs(np.mean(im1) - np.mean(im2))


def find_adjacent(target, all_coords, y_gap=20, x_tolerance=20):
    """
    Takes in the coords of the current center, and finds the centers that are above and below it.
    If it is an edge, then one of the returned values is (0,0).
    :param target:
    :param all_coords:
    :param y_gap:
    :param x_tolerance:
    :return:
    """
    cx, cy = target
    above = None
    below = None

    for (ox, oy) in all_coords:
        if (ox, oy) == (cx, cy):
            continue  # Skip self

        if abs(ox - cx) > x_tolerance:
            continue  # so that we don't mistakenly choose a center in the opposite column

        dy = oy - cy

        if dy < 0 and abs(dy) >= y_gap:
            if (above is None) or (cy - oy < cy - above[1]):
                above = (ox, oy)
        elif dy > 0 and abs(dy) >= y_gap:
            if (below is None) or (oy - cy < below[1] - cy):
                below = (ox, oy)

    # Return (0, 0) for edges
    return above if above else (0, 0), below if below else (0, 0)


def check_adjacent(above, below, score):
    """
    Checks the status of adjacent centers, and boosts the score accordingly.
    :param above:
    :param below:
    :param score:
    :return:
    """
    boost = 3
    if center_status[above]:
        boost -= 1
    if center_status[below]:
        boost -=1
    return  round(score*boost, 2)


# MAIN:

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
            # sees if there is a difference between the last frame and current frame, more efficient


    if frame_num % interval == 0:
        if last_frame is None:
            arr_opt = range(len(slots))
        else:
            arr_opt = [n for n in np.argsort(diff) if diff[n] / np.amax(diff) > 0.4]
            # removes outliers, improving performance

        last_frame = frame.copy()

        #creating rectangle for each spot
        for i in arr_opt:
            x1, y1, w, h, center = slots[i]

            x2, y2 = x1+w, y1+h
            space = frame[y1:y2, x1:x2, :]
            status = parking_space(space)
            center_status[center] = status
            dist = abs(center[1] - top_y) * (1 / 3)
            if center_status[center]:
                score[center] = dist


    for i, slot in enumerate(slots):
        x1, y1, w, h, center = slots[i]
        x2, y2 = x1+w, y1+h
        cx, cy = center




        above, below = find_adjacent(center, center_spots)

        # below code is to check if adjacent centers are being detected properly
        # ax, ay = above
        # bx, by = below

        # cv2.line(frame, (int(ax), int(ay)), (int(cx), int(cy)), (0, 0, 0), 2)
        # cv2.line(frame, (int(cx), int(cy)), (int(bx), int(by)), (0, 0, 0), thickness=2)

        if center_status[center]:
            final_score[center] = check_adjacent(above, below, score[center])
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        best_cx, best_cy = min(final_score, key=final_score.get)
        label = f'{i}, {final_score[center]}'
        frame = cv2.circle(frame, (int(cx), int(cy)), radius=3, color=(128, 0, 128), thickness=2)
        cv2.putText(frame, label, (int(cx), int(cy)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.4, color=(0, 255, 0), thickness=2)

        # plotting roads

        max_cy = best_cy+10 # so that line stops at y-value of best parking spot
                            # (+10 to allow for not perfectly aligned points)


        for i, road in enumerate(roads):
            _, _, _, _, (cx_r, cy_r) = road
            if cy_r <= max_cy:
                cv2.circle(frame, (int(cx_r), int(cy_r)), 5, (0, 255, 255), -1)
                cv2.putText(frame, str(i), (int(cx_r) + 5, int(cy_r) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)


        for i in range(len(roads) - 1):
            _, _, _, _, (cx1_r, cy1_r) = roads[i]
            _, _, _, _, (cx2_r, cy2_r) = roads[i + 1]
            if cy1_r <= max_cy and cy2_r <= max_cy:
                last_cx, last_cy = cx2_r, cy2_r
                cv2.line(frame, (int(cx1_r), int(cy1_r)), (int(cx2_r), int(cy2_r)), (128, 0, 128), 5)

    cv2.line(frame, (int(last_cx), int(last_cy)), (int(best_cx), int(best_cy)), (128, 0, 128), 5)


    frame_num += 1

    if not ret:
        print("End of video or failed to read frame.")
        break
    occupied = sum(value for value in center_status.values())
    total = len(center_spots)
    cv2.putText(frame, f'Available Spots: {occupied-1}/{total}', (0, 20), # -1 because (0,0)
                cv2.FONT_ITALIC, 0.5, (255, 0, 0), 2)
    writer.write(frame)
    cv2.imshow('Parking Footage', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'): # basically closes the frame when we hit q
        break
    # waits 25ms for a keypress (q) -> closes the window

vid.release()
writer.release()
cv2.destroyAllWindows()