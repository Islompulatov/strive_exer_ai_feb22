class Pyramid():
    def __init__(self,num):
        self.num = num
        
    def pyramid(self):
        i =0
        for a in range(1,self.num+1):
            for b in range(1,(self.num-a)+1):
                print(end = " ")
        
        
            while i != (a-1): 
                print("#", end = "") 
                i += 1
            i = 0   
            print()  
            

build_pyr = Pyramid(10)
build_pyr.pyramid() 