from contextlib import contextmanager
from typing import Any
from decorator import decorator


class NaiveContextManager(object):

    def __init__(self, ):
        self.connection_pool = []

    def __len__(self):
        return len(self.connection_pool)

    def __enter__(self):
        print("enter context manager scope")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("exit context manager scope")

@contextmanager
def generator_context_manager():
    print("enter generator context manager scope")
    yield [1,2,3]
    print("exit generator context manager scope")


class ContextManager(object):

    def __init__(self, func, *args, **kwargs):
        self.connection_pool = []
        self.__generator = func(*args, **kwargs)

    def __enter__(self):
        try:
            return next(self.__generator)
        except StopIteration:
            pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        try: 
            next(self.__generator)
        except StopIteration:
            pass

def generator():
    print("enter manual generator context manager scope")
    yield [1,2,3]
    print("exit manual generator context manager scope")

@decorator(ContextManager)
def generator_with_decorator():
    print("enter generator with decorator context manager scope")
    yield [1,2,3]
    print("exit generator with decorator context manager scope")


class Model(object):

    class Module(object):
        def __init__(self, name,):
            self.name = name
            self._mode = "Train"

        def eval(self,):
            self._mode = "Eval"
        
        def train(self,):
            self._mode = "Train"

        def __call__(self, *args: Any, **kwds: Any) -> Any:
            print(f"{self.name} is in {self._mode} mode")

    def __init__(self, name):
        self.name = name
        self.learning_rate = 0.01
        self.q_net = self.Module(name="Q")
        self.target_q_net = self.Module(name="Target_Q")

    @contextmanager
    def eval_mode(self,):
        torch_modules = []
        for  member in self.__dict__.values():
            if isinstance(member, self.Module):
                torch_modules.append(member)
        for member in torch_modules:
            member.eval()
        yield
        for member in torch_modules:
            member.train()

    def update(self,):
        
        with self.eval_mode():
            self.target_q_net()

        self.q_net()
        



if __name__ == "__main__":

    # with NaiveContextManager() as ctx:
    #     print(f"{len(ctx)=}")
    #     print("processing in context manager scope")

    # with generator_context_manager() as ctx:
    #     print(f"{len(ctx)=}")
    #     print("processing in context manager scope")

    # with ContextManager(func=generator) as ctx:
    #     print(f"{len(ctx)=}")
    #     print("processing in manual generator context manager scope")

    # with generator_with_decorator() as ctx:
    #     print(f"{len(ctx)=}")
    #     print("processing in generator with decorator context manager scope")


    model = Model(name="DQN")
    model.update()
    