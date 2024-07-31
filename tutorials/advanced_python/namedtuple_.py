from collections import namedtuple


class Point3d(namedtuple('Point3d', 'x y z')):
    __slots__ = ()
    def __new__(cls, x, y, z=0):
        return super().__new__(cls, x, y, z)
    


if __name__ == '__main__':
    p = Point3d(1, 2, 3)
    print(p)
    q = Point3d(0, -1)
    print(q)
    print(p.__slots__)
    # print(p.__dict__)