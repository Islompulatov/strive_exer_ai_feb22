def nested_nest():
    for a in range(1,9):
        for b in range(0,9):
            for c in range(0,9):
                for d in range(1,9):
                    if 4*(a*1000+b*100+c*10+d) == (a+b*10+c*100+d*1000):
                        print("A= "+str(a)+"\nB= "+str(b),"\nC= "+str(c),"\nD= "+str(d))
nested_nest() 
 

print([str(a)+str(b)+str(c)+str(d)  for a in range(1,9) for b in range(1,9) for c in range(1,9) 
for d in range(1,9) if 4*(a*1000+b*100+c*10+d) == (a+b*10+c*100+d*1000)]) 