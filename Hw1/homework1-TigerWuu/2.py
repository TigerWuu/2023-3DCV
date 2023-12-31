import sys
import numpy as np
import cv2 as cv
import cv3D as c3

WINDOW_NAME = 'window'


def on_mouse(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        param.append([x, y])  


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('[USAGE] python3 mouse_click_example.py [IMAGE PATH]')
        sys.exit(1)

    img = cv.imread(sys.argv[1])
    # scale = 5
    # warpimg = np.zeros([int(img.shape[0]/scale),int(img.shape[1]/scale),3],dtype=np.uint8) # assign the data type of this array to uint 8 for showing this image conveniently

    points1 = []
    # points2 = [[0,0], [warpimg.shape[1]-1,0], [warpimg.shape[1]-1,warpimg.shape[0]-1], [0, warpimg.shape[0]-1]] # camera coordinates
    cv.namedWindow(WINDOW_NAME)
    cv.setMouseCallback(WINDOW_NAME, on_mouse, points1)
    while True:
        img_ = img.copy()
        for i, p in enumerate(points1):
            # draw points on img_
            cv.circle(img_, tuple(p), 20, (0, 0, 255), -1)
        cv.imshow(WINDOW_NAME, img_)

        key = cv.waitKey(20) % 0xFF
        if key == 27: break # exist when pressing ESC

    cv.destroyAllWindows()
    points1 = np.array(points1)
    # using length and width of the original image
    x_length = np.linalg.norm(points1[0]-points1[1]) 
    y_length = np.linalg.norm(points1[0]-points1[3]) 
    warpimg = np.zeros([int(y_length),int(x_length),3],dtype=np.uint8) # assign the data type of this array to uint 8 for showing this image conveniently
    points2 = [[0,0], [x_length-1,0], [x_length-1,y_length-1], [0, y_length]] # camera coordinates
    points2 = np.array(points2)
    
    # print(points1) 
    # print(points2)
    print('{} Points added'.format(len(points1)))

    H_nor = c3.findHomographyNDLT(points1 , points2, 4, method=0, threshold=3, iteration=2000, numPick=4)
    warpimage = c3.warpPerspective(H_nor,img, warpimg, method="NN")
    cv.imshow("Warpimage", warpimage)
    cv.waitKey(0)

