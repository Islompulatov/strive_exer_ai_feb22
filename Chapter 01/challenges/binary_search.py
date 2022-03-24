def sorted_func(list2):
    result = 0
    for i in list2:
        result += i
    return result
sorted([1,5,3,7,93,6])
def merge_func(list1, val):
    n = len(list1)
    left = 0
    right = n-1
    while left <= right:
        middle = (left+right)//2
        if val < list1[middle]:
            right = middle-1
        elif val > list1[middle]:
            left = middle+1
        else:
            return middle  
merge_func([1,3,5,8,9,1,5], 5)