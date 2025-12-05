import cv2
import numpy as np

def mser(image_bgr):
    return mser_pyramid(
        image_bgr
    )


def mser_pyramid(
    image_bgr,
    scales=[1.0],
    min_area=60,
    max_area=8000,
    delta=5,
    max_variation=0.25,
    min_diversity=0.2,
    nms_iou=0.3,
    contain_iou=0.8,
):

    h_img, w_img = image_bgr.shape[:2]
    gray_full = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    all_boxes = []

    for s in scales:
        # resize image for this scale
        if s == 1.0:
            gray = gray_full
        else:
            gray = cv2.resize(
                gray_full, None,
                fx=s, fy=s,
                interpolation=cv2.INTER_LINEAR
            )

        # scale-aware area 
        scaled_min_area = int(min_area * (s ** 2))
        scaled_max_area = int(max_area * (s ** 2))

        mser = cv2.MSER_create(
            delta,
            scaled_min_area,
            scaled_max_area,
            max_variation,
            min_diversity,
            200,
            1.01,
            0.003,
            5
        )

        regions, _ = mser.detectRegions(gray)

        for pts in regions:
            x, y, w, h = cv2.boundingRect(pts)

            # back-project to original scale
            if s != 1.0:
                x = int(x / s)
                y = int(y / s)
                w = int(w / s)
                h = int(h / s)

            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = max(1, min(w, w_img - x))
            h = max(1, min(h, h_img - y))

            # basic filtering: very tiny / weird aspect
            if w < 5 or h < 5:
                continue

            # ratio is abnormal
            aspect = w / float(h)
            if aspect < 0.2 or aspect > 5.0:
                continue

            all_boxes.append((x, y, w, h))
    if not all_boxes:
        return []
    boxes_nms = nms_boxes(all_boxes, iou_thresh=nms_iou, contain_thresh=contain_iou)
    return boxes_nms


def nms_boxes(boxes, iou_thresh=0.2, contain_thresh=0.8):
    boxes_np = np.array(boxes)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = boxes_np[:, 0] + boxes_np[:, 2] # w
    y2 = boxes_np[:, 1] + boxes_np[:, 3] # h
    
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = np.argsort(y1)[::-1] # from bottom
    sorted_boxes = boxes_np[order]
    
    keep_idx = []
    while order.size > 0:
        current = order[0]
        keep_idx.append(current)

        if order.size == 1:
            break
        # cal inter
        xx1 = np.maximum(x1[current], x1[order[1:]])
        yy1 = np.maximum(y1[current], y1[order[1:]])
        xx2 = np.minimum(x2[current], x2[order[1:]])
        yy2 = np.minimum(y2[current], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        inter = w*h
        iou = inter / (areas[current] + areas[order[1:]] - inter)
        contain = inter / (np.minimum(areas[current], areas[order[1:]]))
        idxs = np.where((iou <= iou_thresh) & (contain <= contain_thresh))[0]
        
        order = order[idxs + 1]

    res_boxes = boxes_np[keep_idx].astype(int).tolist()
    return res_boxes


# def sliding_window(image, step_size, window_size):
#     """
#     Basic sliding window generator.
#     """
#     for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
#         for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
#             yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])