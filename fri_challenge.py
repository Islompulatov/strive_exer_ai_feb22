def pyramid(num):
    for a in range(num):
        print(a*"#")
        
pyramid(6)        



def pyramid1(num):
    for a in range(num):
        for b in range(a+1):
            print(b, end = " ")
        print("\n")        
             
pyramid1(6)             