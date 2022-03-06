def pyramid(num):
    i =0
    for a in range(1,num+1):
        for b in range(1,(num-a)+1):
            print(end = " ")
        while i != (a-1): 
            print("#", end = "") 
            i += 1
        i = 0   
        print()  
            

pyramid(5)        