class Base:
    def __init__(self):
        self.list_object = []

    def __call__(self, list_object):
        self.list_object = list_object

    def __getitem__(self, index):
        return self.list_object[index]

    def __len__(self):
        return len(self.list_object)
