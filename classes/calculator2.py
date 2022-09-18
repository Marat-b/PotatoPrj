from classes.calculator import Calculator


class Calculator2(Calculator):
    def __init__(self, metrics: []):
        super().__init__(metrics)
        self.potato_class = {}

    def add_class(self, id_entity: str, class_number: int):
        # self.redis.zadd('potato', {id: size})
        self.potato_class[id_entity] = class_number

    def count_classes(self) -> {}:
        """

        :return: {'strong': 2, sick_name: 0, sick_name: 0, .. and so on}
        :rtype:
        """
        sorted_potato = {}
        # print(f'potato_class={self.potato_class}')
        for key in self.potato_class.keys():
            value = self.potato_class[key]
            if value in sorted_potato:
                sorted_potato[value] += 1
            else:
                sorted_potato[value] = 1
        return sorted_potato