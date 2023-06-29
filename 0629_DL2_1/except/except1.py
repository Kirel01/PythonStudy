(x,y) = (2,0)

try:
    z = x/y
except ZeroDivisionError:
    print("0으로 나누는 예외발생함")

try:
    z = x/y
except ZeroDivisionError as e:
    print(e)    