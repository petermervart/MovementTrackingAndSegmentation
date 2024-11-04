import cv2
import numpy as np
import imageio


def nothing(x):
    pass

#  function for resizing images
def resize_image_ratio(img, ratio_percent):
    width = int(img.shape[1] * ratio_percent / 100)
    height = int(img.shape[0] * ratio_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return img

#  function for gamma correction
def gamma_correction(img, gamma):
    gamma = 1 / gamma

    table = [((i / 255) ** gamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(img, table)

#  function for LK
def video_LK(pathfile):

    colors = []

    capture = cv2.VideoCapture(pathfile)

    prev_frame = capture.read()[1]

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    mask = np.zeros_like(prev_frame)

    frame_id = 0

    points_prev = []

    images = []

    subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=False)  # init MOG2

    while (True):

        ret, current_frame = capture.read()  # get current frame from video
        if not ret:
            break

        if frame_id == 0 or len(points_prev) < 20:  # if its first frame or we have less than 20 active points
            sub_mask = subtractor.apply(current_frame, None, 0.002)  # apply subtractor to get mask of moving objects and apply morphological operations
            ret, sub_mask = cv2.threshold(sub_mask, 30, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
            sub_mask = cv2.morphologyEx(sub_mask, cv2.MORPH_OPEN, kernel=kernel, iterations=1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
            sub_mask = cv2.dilate(sub_mask, kernel=kernel, iterations=1)
            cv2.imshow('mask', resize_image_ratio(sub_mask, 70))  # show mask
            if len(points_prev) == 0:  # if points from previous iteration are empty
                points_prev = cv2.goodFeaturesToTrack(prev_frame_gray, mask=sub_mask, maxCorners=50, qualityLevel=0.05, minDistance=4, blockSize=7)

            else:  # if there are some existing points we add them with the new ones
                points_new = cv2.goodFeaturesToTrack(prev_frame_gray, mask=sub_mask, maxCorners=50, qualityLevel=0.05, minDistance=4, blockSize=7)
                if(points_new is not None):
                    points_prev = np.append(points_new, points_prev, axis=0)

        else:  # update subtractor
            sub_mask = subtractor.apply(current_frame, None, 0.002)

        current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if points_prev is not None and len(points_prev) > 0:  # apply OpticalFlowPyrLK
            points_current, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, current_frame_gray, points_prev, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        if points_current is not None and points_prev is not None:  # filter points
            status = status.flatten()
            prev_filtered = points_prev[status == 1]
            current_filtered = points_current[status == 1]

        curr_color_index = 0

        new_current_filtered = []  # new filtered points

        for i, (prev, current) in enumerate(zip(prev_filtered, current_filtered)):  # draw lines
            x_curr, y_curr = current.ravel()
            x_prev, y_prev = prev.ravel()
            x_prev = int(x_prev)
            x_curr = int(x_curr)
            y_prev = int(y_prev)
            y_curr = int(y_curr)
            if x_prev != x_curr or y_prev != y_curr:  # at least one pixel move
                if curr_color_index > len(colors) - 1:  # if we dont have enough colors create another
                    colors.append(np.random.choice(range(256), size=3).tolist())
                mask = cv2.line(mask, (int(x_curr), int(y_curr)), (int(x_prev), int(y_prev)), colors[curr_color_index], 2)  # append valid point
                new_current_filtered.append(current_filtered[i])
                curr_color_index += 1

        new_current_filtered = np.array(new_current_filtered)  # np array from list

        current_frame_mask = cv2.add(current_frame, mask)  # draw lines on frame

        cv2.imshow('frame', resize_image_ratio(current_frame_mask, 70))

        prev_frame_gray = current_frame_gray.copy()  # change prev frame to current and move to another iteration

        points_prev = new_current_filtered

        if (frame_id % 7 == 0):
            images.append(cv2.cvtColor(resize_image_ratio(sub_mask, 20), cv2.COLOR_BGR2RGB))

        frame_id += 1

        k = cv2.waitKey(20) & 0xFF
        if k == 27:
            break

    imageio.mimsave('video.gif', images, fps=24)

    cv2.destroyAllWindows()

#  function for dense optical
def video_farneback(pathfile):
    cap = cv2.VideoCapture(cv2.samples.findFile(pathfile))  # get first frame
    ret, frame1 = cap.read()
    frame1 = resize_image_ratio(frame1, 50)
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    images = []

    frame_id = 0

    while (1):
        ret, frame2 = cap.read()

        if not ret:
            print('No frames grabbed!')
            break

        frame2 = gamma_correction(resize_image_ratio(frame2, 50), 1.8)  # apply gamma corection for night video

        next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale=0.8, levels=3, winsize=15, iterations=5, poly_n=5, poly_sigma=1.1, flags=0)  # apply Farneback for prev and current frame
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 2] = cv2.normalize(mag, None, 10, 255, cv2.NORM_MINMAX)
        hsv[..., 0] = ang * 180 / np.pi / 2

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)  # convert hsv to gray for creating mask

        cv2.imshow('frame_mask_bad', resize_image_ratio(gray, 70))

        gray = cv2.GaussianBlur(gray, (5, 5), 0)  # apply blur

        cv2.imshow('frame_mask_blur', resize_image_ratio(gray, 70))
        _, gray = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)  # threshold to remove noise

        cv2.imshow('frame_mask_thresh', resize_image_ratio(gray, 70))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # apply opening
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel=kernel, iterations=1)

        cv2.imshow('frame_mask_opening', resize_image_ratio(gray, 70))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # apply closing
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel=kernel, iterations=1)

        cv2.imshow('frame_mask_closing', resize_image_ratio(gray, 70))

        contours = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]  # find external contours
        bounding_boxes = []
        for i, c in enumerate(contours):
            contour_area = cv2.contourArea(c)
            if contour_area > 800:  # filter contours based on area
                con_poly = cv2.approxPolyDP(c, 3, True)  # generate poly
                bounding_boxes.append(cv2.boundingRect(con_poly))  # add bounding box for valid contour

        if bounding_boxes is not None:
            for box in bounding_boxes:
                cv2.rectangle(frame2, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), (0, 255, 00), 2)  # draw rectangle to frame

        cv2.imshow('frame2', frame2)

        if(frame_id % 7 == 0):
            images.append(cv2.cvtColor(resize_image_ratio(frame2, 40), cv2.COLOR_BGR2RGB))

        frame_id += 1

        k = cv2.waitKey(0) & 0xff
        if k == 27:
            break

        prvs = next

    imageio.mimsave('video.gif', images, fps=24)
    cv2.destroyAllWindows()

#  function for grabcut
def grabcut(pathfile, mask_pathfile):
    image = cv2.imread(pathfile)

    mask_init = cv2.imread(mask_pathfile, 0)  # get init mask created by hand

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  # apply erosion for more realistic mask (the hand mand mask was too good)
    mask_init = cv2.erode(mask_init, kernel=kernel, iterations=1)

    mask = np.zeros(image.shape[:2], np.uint8)  # create mask for grabcut 1 - definetly foreground, 2 - possible background
    mask[mask_init == 255] = 1
    mask[mask_init == 0] = 2

    backgroundModel = np.zeros((1, 65), np.float64)  # init fg and bg
    foregroundModel = np.zeros((1, 65), np.float64)

    cv2.grabCut(image, mask, None, backgroundModel, foregroundModel, 3, cv2.GC_INIT_WITH_MASK)  # apply grabcut

    mask_final = np.where((mask == 0 ) | (mask == 2), 0, 1).astype('uint8')  # get mask with only foreground (filtering image)

    mask_obvious_foreground = np.where((mask == 1), 1, 0).astype('uint8')  # get obvious foreground mask

    mask_possible_foreground = np.where((mask == 3), 1, 0).astype('uint8')  # get possible foreground mask

    image_masked = image * mask_final[:, :, np.newaxis]  # apply mask to image

    contours = cv2.findContours(mask_obvious_foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]  # get and draw sure foreground contours to image

    image = cv2.drawContours(image, contours, -1, (0, 0, 255), 1)

    contours = cv2.findContours(mask_possible_foreground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)[0]  # get and draw possible foreground contours to image

    image = cv2.drawContours(image, contours, -1, (0, 255, 0), 1)

    cv2.imshow('image', resize_image_ratio(image, 200))

    cv2.imshow('image masked', resize_image_ratio(image_masked, 200))

    cv2.waitKey(0) & 0xff


def main():
    #video_LK('video_samples/fourway.avi')
    video_farneback('video_samples/fourway.avi')
    #grabcut('photo_samples/4.png', 'photo_samples/4_mask.png')


if __name__ == "__main__":
    main()