# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import tensorflow as tf;
import numpy as np;
import cv2 as cv;
from PIL import ImageGrab;

while True:
    img_rgb = cv.imread('./pic/coal.PNG')
    img_gray = cv.cvtColor(img_rgb, cv.COLOR_BGR2GRAY)

    img = ImageGrab.grab();
    img2 = np.array(img);
    template = cv.cvtColor(img2,cv.COLOR_BGR2GRAY)
    print(template)
    # w, h = template.shape[::-1]
    w, h = img_rgb.shape[:-1]
    # cv.imshow("screen", template);

    res = cv.matchTemplate(img_gray, template, cv.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where(res >= threshold)
    for pt in zip(*loc[::-1]):
        cv.rectangle(img2, pt, (pt[0] + w, pt[1] + h), (0, 255, 0), 2)
        cv.imshow("screen", cv.cvtColor(img2, cv.COLOR_BGR2RGB));

    if cv.waitKey(1) == 27:
        break;
cv.destroyAllWindows();






