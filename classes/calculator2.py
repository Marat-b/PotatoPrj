from classes.calculator import Calculator


class Calculator2(Calculator):
    def __init__(self, metrics: [], count_frames=0):
        super().__init__(metrics, count_frames=count_frames)
        self.potato_class = {}
        self.count_frames = count_frames

    def add_class(self, id_entity: str, class_number: int):
        # self.redis.zadd('potato', {id: size})
        # self.potato_class[id_entity] = class_number
        if id_entity in self.potato_class:
            self.potato_class[id_entity]['count'] += 1
            self.potato_class[id_entity]['class_number'] = class_number
        else:
            self.potato_class[id_entity] = {'count': 1, 'class_number': class_number}

    def count_classes(self) -> {}:
        """

        :return: {'strong': 2, sick_name: 0, sick_name: 0, .. and so on}
        :rtype:
        """
        sorted_potato = {}
        # print(f'potato_class={self.potato_class}')
        for key in self.potato_class.keys():
            if self.potato_class[key]['count'] > self.count_frames:
                value = self.potato_class[key]['class_number']
                if value in sorted_potato:
                    sorted_potato[value] += 1
                else:
                    sorted_potato[value] = 1
        return sorted_potato