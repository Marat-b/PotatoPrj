import numpy as np
import cv2


def GetFocalPixel(object_width, FOV):
    """
    Get focal length in pixel
    :param object_width: matrix length in pixel
    :param FOV: focal of view in degree
    :return: focal length in pixel
    """
    focal_pixel = (object_width * 0.5) / math.tan((FOV * 0.5 * math.pi) / 180)
    return focal_pixel


###########################################

# focal_pixel_width_back = GetFocalPixel(2560, 93)

#############################################

# vis_frame, body_height, body_width = actions.GetAction2(
#     image, predictor,
#     focal_pixel_width_back, focal_pixel_height_back,
#     5, body_side=actions.BodySide.BACK.value
# )


##############################################

def GetAction2(video_frame, pred_classes=any, focal_pixel_width=any,
               focal_pixel_height=any,
               distance_to_object=any
               ):
    ################### only back bodyshelf ########################################
    # global focal_pixel_width, focal_pixel_height
    vide_image = np.copy(video_frame)

    # print(f'pred_classes={pred_classes}')
    ###########################################
    pred_classes_index = np.arange(len(pred_classes))
    pred_couple = []
    for i, j in zip(pred_classes_index, pred_classes):
        # print(f'i={i}, j={j}')
        pred_couple.append([i, j])
    # print(f'pred_couple={pred_couple}')
    back_bodyshelfs = [back_bodyshelf for back_bodyshelf in pred_couple if back_bodyshelf[1] == body_side]
    # print(f'back_bodyshelfs={back_bodyshelfs}')

    len_back_bodyshelfs = len(back_bodyshelfs)

    if len_back_bodyshelfs > 0:
        for i in range(len_back_bodyshelfs):
            image_mask = outputs['instances'].get('pred_masks')[back_bodyshelfs[i][0]].cpu().numpy()
            image_mask = image_mask.astype(np.uint8)
            image_mask[image_mask > 0] = 255
            # cv2_imshow(image_mask)

            ########### image action ###################################

            ########################## warp action ###############
            contours, box_image = DrawAreaRect2(image_mask, drawCnt=False)
            # print(f'old rect={contours[0][1]}')
            rect = order_points(contours[0][1])
            # print(f'new rect={rect}')
            dst, maxHeight, maxWidth = GetMetrics(rect)
            # print(f'top maxHeight={maxHeight}, maxWidth={maxWidth}')
            # compute the perspective transform matrix and then apply it
            M = cv2.getPerspectiveTransform(rect, dst)
            warped = cv2.warpPerspective(image_mask, M, (maxWidth, maxHeight))
            h, w = warped.shape
            # print(f'h={h}, w={w}')
            # print(f'warped.shape={warped.shape}')
            body_height = GetBodySide(h, distance_to_object, focal_pixel_height)
            body_width = GetBodySide(w, distance_to_object, focal_pixel_width)
            # print(f'body_height={body_height}, body_width={body_width}')
            vide_image = DrawArrowForWarped(video_frame, rect, warped.shape, top_body=False)

            cv2.putText(
                vide_image, 'Height - {}m '.format(round(body_height, 2)), (30, 50),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 255), 2
            )
            cv2.putText(
                vide_image, 'Width - {}m '.format(round(body_width, 2)), (30, 110),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 255), 2
            )
            cv2.putText(
                vide_image, 'Square - {}mm '.format(round((body_width * body_height), 2)), (30, 170),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (0, 0, 255), 2
            )
            # cv2_imshow(top_bodyshelf_image)
            ###################### end warp action ####################
        return vide_image, body_height, body_width
    else:
        return vide_image, 0, 0


###########################################################################

def GetBodySide(height_pixel, distance_to_object, focal_pixel):
    body_side = (height_pixel * distance_to_object) / focal_pixel
    return body_side


def GetMetrics(rect):
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array(
        [
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32"
    )
    return dst, maxHeight, maxWidth


def DrawAreaRect2(gray_image, drawCnt=False):
    """
    DrawAreaRect2 for warp action
    :param color_image:
    :param drawCnt:
    :return:
    """
    # gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    # plt.title('thresh')
    # plt.imshow(thresh)
    # plt.show()
    contours, hiearchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f'contours={np.array(contours).shape}')
    # print(f'contours={contours}')
    finalContours = []
    for i, cnt in enumerate(contours):
        # cnt = contours[ind]
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        ####################################
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.05 * peri, True)
        finalContours.append((box, approx))
        # print(f'cnt={i}, box={box}')
        if drawCnt:
            print(f'approx={approx}')
            bbox = cv2.boundingRect(approx)
            # print(f'bbox={bbox}')
            gray_image = cv2.polylines(gray_image, approx, True, (200, 0, 0), 20)
            # color_image = cv2.circle(color_image, (bbox[0], bbox[1]), 5, (200, 0, 0), -1)

            gray_image = cv2.drawContours(gray_image, [box], 0, (0, 0, 255), 5)
            cv2_imshow(gray_image, title='DrawAereaRect2')
            # ellipse = cv2.fitEllipse(cnt)
            # color_image = cv2.ellipse(color_image, ellipse, (0, 255, 0), 2)

    return finalContours, gray_image


def order_points(pts):
    # print(f'len(pts)={(pts.shape)[0]}')
    if (pts.shape)[0] == 4:
        pts = pts.reshape((4, 2))
        # sort the points based on their x-coordinates
        xSorted = pts[np.argsort(pts[:, 0]), :]

        # grab the left-most and right-most points from the sorted
        # x-roodinate points
        leftMost = xSorted[:2, :]
        rightMost = xSorted[2:, :]

        # now, sort the left-most coordinates according to their
        # y-coordinates so we can grab the top-left and bottom-left
        # points, respectively
        leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
        (tl, bl) = leftMost

        # now that we have the top-left coordinate, use it as an
        # anchor to calculate the Euclidean distance between the
        # top-left and right-most points; by the Pythagorean
        # theorem, the point with the largest distance will be
        # our bottom-right point
        D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
        (br, tr) = rightMost[np.argsort(D)[::-1], :]

        # return the coordinates in top-left, top-right,
        # bottom-right, and bottom-left order
        return np.array([tl, tr, br, bl], dtype="float32")
    else:
        return np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype="float32")


def show_area(gray_image):
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    # plt.title('thresh')
    # plt.imshow(thresh)
    # plt.show()
    contours, hiearchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(f'contours={np.array(contours).shape}')
    # print(f'contours={contours}')
    finalContours = []
    width = 0
    for i, cnt in enumerate(contours):
        # cnt = contours[ind]
        rect = cv2.minAreaRect(cnt)
        sides = rect[1]
        width = sides[0] if sides[0] > sides[1] else sides[1]
        print(f'show_areas width={width}')
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # print(f'show_area box={box}')
        # points = np.array(box)
        # color = (255, 0, 0)
        # thickness = 2
        # isClosed = False
        # image = gray_image.copy()
        # # drawPolyline
        # image = cv2.polylines(image, [points], isClosed, color, thickness)
        # # cv2_imshow(image)
    return int(width)
