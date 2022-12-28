import numpy as np

def add(atuple, btuple):
    a, adot = atuple
    b, bdot = btuple
    return (a+b, adot + bdot)

def subtract(atuple, btuple):
    a, adot = atuple
    b, bdot = btuple
    return (a-b, adot-bdot)


def mul(atuple, btuple):
    a, adot = atuple
    b, bdot = btuple
    return (a*b, adot*b+a*bdot)

def div(atuple, btuple):
    a, adot = atuple
    b, bdot = btuple
    return (a/b, (adot*b - bdot*a)/(b*b))

def exp(atuple):
    a, adot = atuple
    return np.exp(a), np.exp(a)*adot

def sin(atuple):
    a, adot = atuple
    return np.sin(a), np.cos(a)*adot


def myfunc(x1, x2):
    a = div(x1, x2)
    b = exp(x2)
    return mul(subtract(add(sin(a), a), b), subtract(a, b))

print(myfunc( (1.5, 1.0), (0.5, 1.0)))
