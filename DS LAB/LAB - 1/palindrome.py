x=int(input("Enter a number: "))
if str(x) == str(x)[::-1]:
    print("The number is a palindrome") 
else:
    print("The number is not a palindrome")