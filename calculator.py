# import redis


class Calculator:
    def __init__(self):
        self.potato = {}
        # self.redis = redis.Redis()
        # self.redis.set('potato_strong', 0)
        # self.min_size = min_size
        # self.max_size = max_size

    def add(self, id_entity: str, size: float):
        # self.redis.zadd('potato', {id: size})
        self.potato[id_entity] = size

    def count(self, min_size: float, max_size: float) -> float:
        # if self.min_size < size < self.max_size:
        #     self.redis.incr('potato_strong')
        # self.redis.zcount('potato', min_size, max_size)
        count = 0
        for key in self.potato.keys():
            if min_size <= self.potato[key] < max_size:
                count += 1
        return count


if __name__ == '__main__':
    c = Calculator()
    c.add('1', 0.3)
    c.add('1', 0.1)
    c.add('2', 0.4)
    c.add('3', 0.5)
    count = c.count(0.2, 0.5)
    print(count)
