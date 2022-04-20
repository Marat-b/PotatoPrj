import cv2
import random


class Drawer:
    def __init__(self):
        self._bbox = None
        self._calculator = None
        self._entity = None
        self._identity = None
        self._mask = None
        self._measurement = None
        self._outputs = None
        self._color_list = []
        self._color_index = 999
        self._class_names = None
        for j in range(self._color_index + 1):
            self._color_list.append(
                (int(random.randrange(255)), int(random.randrange(255)), int(random.randrange(255)))
            )

    def add_bbox(self, bbox):
        self._bbox = bbox
        return self

    def add_calculator(self, calculator):
        self._calculator = calculator
        return self

    def add_class_names(self, class_names):
        self._class_names = class_names
        return self

    def add_entity(self, entity):
        self._entity = entity
        return self

    def add_identity(self, identity):
        self._identity = identity
        return self

    def add_mask(self, mask):
        self._mask = mask
        return self

    def add_measurement(self, measurement):
        self._measurement = measurement
        return self

    def outputs(self, img, outputs):
        image = img.copy()
        self._outputs = outputs
        bboxes = outputs[:, :4]
        self._bbox(bboxes)
        # print(f'outputs[:, -3]={outputs[:, -3]}')
        self._identity(outputs[:, -3])
        self._entity(outputs[:, -2])
        self._mask(outputs[:, -1])
        t_size = []
        # len_bbox = len(self._bbox)
        # print(f'len_bbox={len_bbox}')
        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = [int(i) for i in box]
            # print(f'x1={x1}, x2={x2}, y1={y1}, y2={y2}')
            width = float('{:.2f}'.format(self._measurement.get_width_meter(self._mask[i])))
            self._calculator.add(self._identity[i], width)
            color = self._color_list[int(self._identity[i] % self._color_index)]
            label = '{}-{:d} w={}m'.format(self._class_names[self._entity[i]], self._identity[i], str(width))
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(image, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(image, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        item_sorted = self._calculator.count()
        for i, key in enumerate(item_sorted.keys()):
            cv2.putText(image, 'amount of {}={}'.format(key, str(item_sorted[key])),
                        (5, 5 + t_size[1] + 4 + i*(t_size[1])), cv2.FONT_HERSHEY_PLAIN, 2, [0, 0, 255], 2)
        return image
