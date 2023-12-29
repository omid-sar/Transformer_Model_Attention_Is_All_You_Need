def f1(func):
    def wrapper():
        print('started')
        func()
        print('ended')
    return wrapper    

def f():
    print("Hello")

x=3

f1()
f1(f)()


import torch

print(dir(torch))
help(torch)