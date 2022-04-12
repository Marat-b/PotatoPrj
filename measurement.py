import numpy as np

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

focal_pixel_width_back = GetFocalPixel(2560, 93)

#############################################

vis_frame, body_height, body_width = actions.GetAction2(
    image, predictor,
    focal_pixel_width_back, focal_pixel_height_back,
    5, body_side=actions.BodySide.BACK.value
)


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
