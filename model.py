import torch 
import torch.nn as nn

import time

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} ran in: {end - start} sec")
        return result
    return wrapper

@timer
def some_function():
    time.sleep(5)

some_function()  # Outputs the execution time of some_function



def f1():
    print('call f1')

f1()
print(f1)
f1

def f2(f):
    f()

f2(f1)