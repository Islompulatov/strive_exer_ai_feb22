def pyramid(num):
    for a in range(num):
        print(a*"#")
        
pyramid(6)        



from sympy import prime


def pyramid1(num):
    for a in range(num):
        for b in range(a+1):
            print("#", end = "")
        print("\n")
             
pyramid1(6)             