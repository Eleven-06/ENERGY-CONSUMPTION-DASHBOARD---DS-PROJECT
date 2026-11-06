x,y,z = "Apple", "Banana", "Cherry"
print(x,y, z)
x=y=z = "Orange"
print(x,y,z)
age = 20
print(age)
height = 5.5
print(height)
var_name = input("Enter your name: ")
print(var_name) # This is a comment
x=1
y=2.8
z=1j
print(type(x))
print(type(y))
print(type(z))
x=2.8
y=int(x)
print(y)
b="KLH BOWREMPET"
print(b[2:5])
print(b[:8])
print(b[5:])
print(b[-5:-2])
a="Hello, World!"
print(a.lower())
print(a.upper())
print(a.capitalize())
print(len(a))
print(a.title())
print(a.swapcase())
a="kl"
b="hyd"
c=a+b
print(c)
c=a+""+b
print(c)
txt1="my name is {fname}, I'm {age}".format(fname="John", age=36)
txt2="my name is {0}, I'm {1}".format("John",36)
txt3="my name is {}, I'm {}".format("John",36)
print(txt1)
print(txt2)
print(txt3)
a=33
b=200
if b > a:
    print("b is greater than a")
else:
    print("a is greater than b")