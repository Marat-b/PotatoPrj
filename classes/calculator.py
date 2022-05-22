# import redis


class Calculator:
    def __init__(self, metrics: []):
        self.potato = {}
        self.metrics = metrics
        # self.redis = redis.Redis()
        # self.redis.set('potato_strong', 0)
        # self.min_size = min_size
        # self.max_size = max_size

    def add(self, id_entity: str, size: float):
        # self.redis.zadd('potato', {id: size})
        self.potato[id_entity] = size

    def count(self) -> {}:
        # if self.min_size < size < self.max_size:
        #     self.redis.incr('potato_strong')
        # self.redis.zcount('potato', min_size, max_size)
        sorted_potato = {}
        # count = 0

        for key in self.potato.keys():
            for metrics in self.metrics:
                name = metrics[0]
                min_size = metrics[1]
                max_size = metrics[2]
                if min_size <= self.potato[key] < max_size:
                    if name in sorted_potato:
                        sorted_potato[name] += 1
                    else:
                        sorted_potato[name] = 1
        return sorted_potato


if __name__ == '__main__':
    c = Calculator()
    c.add('1', 0.3)
    c.add('1', 0.1)
    c.add('2', 0.4)
    c.add('3', 0.5)
    count = c.count(0.2, 0.5)
    print(count)
