def strive(num):
    for a in range(num):
        if a%3 == 0 and a%5 == 0:
            print("Strive School")
        elif a%3 == 0:
            print("Strive")
        elif a%5 == 0:
            print("School")
        else:
            print(a)
strive(50)                        