a=int(input("Enter a number: "))
b=int(input("Enter another number: "))
c=int(input("Enter one more number: "))
if a>b and a>c:
    print("The biggest number is:", a)
elif b>a and b>c:
    print("The biggest number is:", b)
else:
    print("The biggest number is:", c)