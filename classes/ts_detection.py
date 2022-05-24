import cv2
import torch
# from detectron2.export.torchscript_patch import patch_instances
# from detectron2.layers.mask_ops import _do_paste_mask
# from detectron2.structures import Boxes
# from torch import Tensor
import numpy as np


class TorchscriptDetection:
    def __init__(self, path_inference, use_cuda=True):
        self.device = 'cuda' if use_cuda else 'cpu'
        # self.fields = {
        #     "proposal_boxes": Boxes,
        #     "objectness_logits": Tensor,
        #     "pred_boxes": Boxes,
        #     "scores": Tensor,
        #     "pred_classes": Tensor,
        #     "pred_masks": Tensor,
        # }
        # assert path_inference is not None
        # with patch_instances(self.fields):
        self.model = torch.jit.load(path_inference)
        self.model.to(self.device)

    def detect(self, image):
        with torch.no_grad():
            out = self.model(
                     torch.as_tensor(image.astype('float32').transpose(2, 0, 1))
            )
        boxes = out[0].numpy()
        classes = out[1].numpy()
        scores = out[3].numpy()
        # pr_masks = self._do_mask(out, image)
        pr_masks = out[2].numpy()
        # pr_masks = out.pred_masks.cpu().numpy()
        # pr_masks = pr_masks.astype(np.uint8)
        # pr_masks[pr_masks > 0] = 255
        bbox_xcycwh, cls_conf, cls_ids, masks = [], [], [], []

        for (box, _class, score, pr_mask) in zip(boxes, classes, scores, pr_masks):
            # print(box)
            x0, y0, x1, y1 = box
            bbox_xcycwh.append([(x1 + x0) / 2, (y1 + y0) / 2, (x1 - x0), (y1 - y0)])
            cls_conf.append(score)
            cls_ids.append(_class)
            masks.append(pr_mask)

        return np.array(bbox_xcycwh, dtype=np.float64), np.array(cls_conf), np.array(cls_ids), np.array(masks)

    # def _do_mask(self, out, image):
    #     N = len(out[2])
    #     chunks = torch.chunk(torch.arange(N, device='cpu'), N)
    #     img_masks = torch.zeros(
    #         N, image.shape[0], image.shape[1], device='cpu', dtype=torch.bool
    #     )
    #     # masks = out.pred_masks[:, 0, :, :]  # shape was [#instances, 1, 28, 28] --> [#instances, 28, 28]
    #     for inds in chunks:
    #         masks_chunk, spatial_inds = _do_paste_mask(
    #             out[2][inds, :, :, :], out[0].tensor[inds], image.shape[0], image.shape[1],
    #             skip_empty=True
    #         )
    #
    #         masks_chunk = (masks_chunk >= 0.5).to(dtype=torch.bool)
    #         img_masks[(inds,) + spatial_inds] = masks_chunk
    #         # print(img_masks.shape)
    #     # print(np.array(np.uint8(img_masks)*255))
    #     return np.array(np.uint8(img_masks)*255)


if __name__ == '__main__':
    tsd = TorchscriptDetection('../weights/model2.ts', use_cuda=False)
    img = cv2.imread('../images/potato.jpg')
    pred_boxes, scores, pred_classes, pred_masks = tsd.detect(img)
    print(f'pred_boxes={pred_boxes}')
    print(f'scores={scores}')
    print(f'pred_classes={pred_classes}')
    print(f'pred_masks={pred_masks[0].shape}')
    # im = tsd.detect2(img)
    # print(im)
    # cv2.imshow('im', im)
    # cv2.waitKey(1000)
