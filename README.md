# Video Processing with Lucas-Kanade Optical Flow and Image Segmentation using GrabCut - Peter Mervart

## Lucas-Kanade optical flow

In solving the Lucas-Kanade algorithm, I used MOG2 to obtain a mask, which I then use to filter only the moving parts of the video, and only in these areas do I search for features using goodFeaturesToTrack. Before applying this mask, I apply opening and dilation to it to eliminate small parts that are not very mobile and to expand the area where features can be searched. I refresh the features whenever the number of active features drops below 20. In such cases, I use MOG2 again to find 50 new features and add them to the other already tracked features (retaining the old features).

```python
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
```


By inactive features, I mean bad or non-moving features, which I eliminate using two techniques:

1. Based on the status of features from calcOpticalFlowPyrLK (features that cannot be found in the new frame).

```python
points_current, status, err = cv2.calcOpticalFlowPyrLK(prev_frame_gray, current_frame_gray, points_prev, None, winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

if points_current is not None and points_prev is not None:  # filter points
    status = status.flatten()
    prev_filtered = points_prev[status == 1]
    current_filtered = points_current[status == 1]
```

2. If they did not move by at least 1 pixel in any direction between the previous and the current frame.

```python
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
```

### Video fourway

#### Mask generated using MOG2 (without morphological operations applied)

![video](https://user-images.githubusercontent.com/55833503/232345237-12d93614-6baf-4f23-96f4-d31ea4625483.gif)

#### Final tracking

![video](https://user-images.githubusercontent.com/55833503/232344017-f3031fbb-07b0-4cbe-9f1c-c2a4a25d5bb4.gif)

### Video crosswalk

#### Mask generated using MOG2 (without morphological operations applied)

![video](https://user-images.githubusercontent.com/55833503/232345097-dacc5035-d2e4-4002-b90a-8c63fd687f4c.gif)

#### Final tracking 

![video](https://user-images.githubusercontent.com/55833503/232344105-75e1e652-f5e2-40cd-9b48-abbfdc5b859a.gif)

### Video night

#### Mask generated using MOG2 (without morphological operations applied)

![video](https://user-images.githubusercontent.com/55833503/232345131-b594417a-aabd-4db1-950a-96d691f01c9a.gif)

#### Final tracking

![video](https://user-images.githubusercontent.com/55833503/232344215-2caa17b0-f119-463f-9611-db1d53506115.gif)


## Farneback

In the Farneback solution, I first generate the flow between the previous and current frame using calcOpticalFlowFarneback. From this flow, I then obtain the magnitude and direction. Based on the magnitude and direction, I create an HSV image highlighting the moving objects. I convert this image to grayscale so that I can use it as a mask for creating bounding boxes.

```python
flow = cv2.calcOpticalFlowFarneback(prvs, next, None, pyr_scale=0.8, levels=3, winsize=15, iterations=5, poly_n=5, poly_sigma=1.1, flags=0)  # apply Farneback for prev and current frame
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv[..., 2] = cv2.normalize(mag, None, 10, 255, cv2.NORM_MINMAX)
hsv[..., 0] = ang * 180 / np.pi / 2     
```

Based on: https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html

Since the images had a significant amount of noise, I tried to eliminate it.

Procedure for processing the grayscale image into a mask and reducing noise:

0. Image before blur

![image](https://user-images.githubusercontent.com/55833503/232347232-faebb1a9-58b6-49f2-ac6c-f1e8f02384ea.png)

1. Applying Gaussian blur (for better thresholding)

![image](https://user-images.githubusercontent.com/55833503/232347246-8163959f-2349-48d6-b458-87751b663609.png)

2. Thresholding to eliminate slowly moving parts and noise

![image](https://user-images.githubusercontent.com/55833503/232347261-c5699893-bd5c-47b8-abb6-8269f0b2002a.png)

3. Applying opening to remove remaining noise

![image](https://user-images.githubusercontent.com/55833503/232347275-656ea2af-209a-49ed-abe0-a60ac06c5192.png)

4. Applying closing to connect points of moving objects (e.g., separating a head from a body)

![image](https://user-images.githubusercontent.com/55833503/232347283-e07b4070-fc34-416c-b78e-04aa7967cd4b.png)

After obtaining the mask, I find the external contours, filter them based on their size, and render them on the current frame of the video.

```python
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
```

### Video fourway

![video](https://user-images.githubusercontent.com/55833503/232344745-3358642e-fdff-4f3f-8b1c-5c085f3e26a3.gif)

### Video crosswalk

![video](https://user-images.githubusercontent.com/55833503/232344907-57a2a9fe-10c1-4ab4-8a8f-21db40e23432.gif)

### Video night with applied gamma correction

![video](https://user-images.githubusercontent.com/55833503/232345001-89d17975-d6ab-4382-b27c-15b504d6b363.gif)

In the videos, you can see that I only partially succeeded in eliminating the noise, and sometimes the noise is labeled as a moving object. I searched for morphological operation values and contour size limits that would be consistent across all videos. Since the person in the fourway video is very small, it is challenging to find values that can capture the small object without also labeling the noise as a moving object.

## Grabcut

In the segmentation using grabcut, I used a manually created init mask (specifically for each image). I applied erosion to these masks for better demonstration of grabcut, as the mask was too perfect and could have been used directly for filtering. From the init mask, I created a mask that contained information for grabcut. Pixels with values of 255 were set as 1 (definite foreground), and pixels with values of 0 were set as 2 (possible background).

```python
mask = np.zeros(image.shape[:2], np.uint8)  # create mask for grabcut 1 - definetly foreground, 2 - possible background
mask[mask_init == 255] = 1
mask[mask_init == 0] = 2
```

After applying grabcut, I created a mask for definite foreground and possible foreground so that I could obtain contours, which I then drew onto the original image. Additionally, I created another mask that combined definite and possible background to filter out the segmented part.

```python
cv2.grabCut(image, mask, None, backgroundModel, foregroundModel, 3, cv2.GC_INIT_WITH_MASK)  # apply grabcut

mask_final = np.where((mask == 0 ) | (mask == 2), 0, 1).astype('uint8')  # get mask with only foreground (filtering image)

mask_obvious_foreground = np.where((mask == 1), 1, 0).astype('uint8')  # get obvious foreground mask

mask_possible_foreground = np.where((mask == 3), 1, 0).astype('uint8')  # get possible foreground mask

image_masked = image * mask_final[:, :, np.newaxis]  # apply mask to image
```

### Legend

```diff
+ Green - possible foreground

- Red - definite foreground
```

### Image 0

#### Init mask

![0_mask](https://user-images.githubusercontent.com/55833503/232345254-e4e66556-866b-494f-975d-7810f693ee6e.png)

#### Image after applying the mask.

![image](https://user-images.githubusercontent.com/55833503/232345277-18a77829-56c9-409e-9d96-441bbade22dc.png)

#### Rendered contours.

![image](https://user-images.githubusercontent.com/55833503/232345282-a38b218f-1df8-4f45-85ca-9b1672d38e68.png)

### Image 1

#### Init mask

![1_mask](https://user-images.githubusercontent.com/55833503/232345257-d53c6c54-7dfa-4c39-8131-e34d30e86de6.png)

#### Image after applying the mask.

![image](https://user-images.githubusercontent.com/55833503/232345300-6935f985-7274-438f-848e-deb0024b4214.png)

#### Rendered contours.

![image](https://user-images.githubusercontent.com/55833503/232345310-2836c5cd-069c-444c-a7a5-0db8a0e60408.png)

### Image 2

#### Init mask

![2_mask](https://user-images.githubusercontent.com/55833503/232345258-788feccb-56e0-4446-adc2-e3d0cf7f03fc.png)

#### Image after applying the mask.

![image](https://user-images.githubusercontent.com/55833503/232345327-541e89db-7984-461b-bd6c-842a1d87243f.png)

#### Rendered contours.

![image](https://user-images.githubusercontent.com/55833503/232345332-d1f8664e-a878-4195-b2c8-e8681af73a62.png)

### Image 3

#### Init mask

![3_mask](https://user-images.githubusercontent.com/55833503/232345260-bbe8d0cd-76b9-4f7b-b044-7431243bcd21.png)

#### Image after applying the mask.

![image](https://user-images.githubusercontent.com/55833503/232345341-aa9a600d-6390-48e6-a0cc-ba4c3b0a8873.png)

#### Rendered contours.

![image](https://user-images.githubusercontent.com/55833503/232345352-1ca33280-2419-4e3f-9253-29c50c8a4ff9.png)

### Image 4

#### Init mask

![4_mask](https://user-images.githubusercontent.com/55833503/232345261-deec5cdf-745a-4578-8419-77338d5c8df4.png)

#### Image after applying the mask.

![image](https://user-images.githubusercontent.com/55833503/232345364-f367ba70-0702-407e-8e6f-45a4964d1bdd.png)

#### Rendered contours.

![image](https://user-images.githubusercontent.com/55833503/232345371-39ee7ba9-6f86-4e16-8d7f-906ee714cabd.png)



