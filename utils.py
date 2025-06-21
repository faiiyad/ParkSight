import cv2
import joblib
import numpy as np
from skimage.transform import resize

model = joblib.load(open("./train_data/cv_t/model.p", "rb"))


def parking_space(src):



    resized = resize(src, (15, 15, 3))
    x = np.array([resized.flatten()])

    y_pred= model.predict(x)

    if y_pred == 0:
        return True
    else:
        return False

def get_parking_boxes(cc):
    (total_labels, label_ids, values, center) = cc

    slots = []
    centers = []

    for i in range(1, total_labels):
        #creates a bounding box
        cx, cy = center[i][0], center[i][1]
        x1 = int(values[i, cv2.CC_STAT_LEFT]*1)
        y1 = int(values[i, cv2.CC_STAT_TOP]*1)
        w = int(values[i, cv2.CC_STAT_WIDTH]*1)
        h = int(values[i, cv2.CC_STAT_HEIGHT]*1)

        slots.append([x1, y1, w, h, (cx, cy)])


    return slots