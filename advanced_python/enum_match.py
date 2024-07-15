from enum import Enum


# class Action(Enum):
#     MOVE_A = (0, (0, 0))
#     MOVE_B = (1, (0, 0))
#     MOVE_C = (2, (0, 0))

#     __match_args__ = ('position')

#     def __eq__(self, other: object) -> bool:
#         if isinstance(other, Action):
#             return self.value == other.value and self.position == other.position
#         return False

#     def __new__(cls, type_name, position):
#         obj = object.__new__(cls)
#         obj._value_ = type_name
#         obj.position = position
#         return obj
    
#     def __call__(self, x, y):
#         self.position = (x, y)
#         return self



class Action(object):
    __match_args__ = ('x', 'y')
    def __init__(self, x, y):
        self.x = x
        self.y = y

class MacroAction(object):
    class MOVE_A(Action): pass
    class MOVE_B(Action): pass



def main():
    actions = [
        MacroAction.MOVE_A(1,2),
        MacroAction.MOVE_B(3,4),
        MacroAction.MOVE_A(5,6),
        MacroAction.MOVE_B(7,8)
    ]
    for a in actions:
        match a:
            case MacroAction.MOVE_A(x, y):
                print(f'Move A to ({x}, {y})')

            case MacroAction.MOVE_B(x, y):
                print(f'Move B to ({x}, {y})')

if __name__ == "__main__":
    main()