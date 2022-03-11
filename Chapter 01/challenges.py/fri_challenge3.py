def duplicate(list):
    list1 = []
    for i in list:
        if list.count(i)>1:
            pass
        else:
            list1.append(i)
    return list1    

list = [1,2,3,4,5,7,2,3,5,4,8,9]
print(duplicate(list))


