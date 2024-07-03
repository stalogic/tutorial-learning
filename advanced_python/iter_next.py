

class IterObject(object):

    def __init__(self, num) -> None:
        self.count = num

    def __iter__(self):
        # return self
        for i in range(self.count):
            yield i
    
    def __next__(self):
        if self.count > 0:
            val = self.count
            self.count -= 1
            return val
        else:
            raise StopIteration
    

if __name__ == "__main__":
    iter_obj = IterObject(5)

    it = iter(iter_obj)
    print(f"{type(iter_obj)=}")
    print(f"{type(it)=}")

    arr = [1, 2, 3, 4, 5]
    it = iter(arr)
    print(f"{type(arr)=}")
    print(f"{type(it)=}")

    # for i in it:
    #     print(i)
    
    # while True:
    #     try:
    #         print(next(iter_obj))
    #     except StopIteration:
    #         break

    it1 = iter(arr)
    it2 = iter(arr)
    while True:
        try:
            print(next(it1))
            print(next(it2))
        except StopIteration:
            break

    iter_obj = IterObject(5)
    it1 = iter(iter_obj)
    it2 = iter(iter_obj)
    while True:
        try:
            print(next(it1))
            print(next(it2))
        except StopIteration:
            break
