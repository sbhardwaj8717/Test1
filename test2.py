fh="APPLE"
lst=list()
for line in fh:
    temp=line.split()
    for i in temp:
        i=ord(i)+3
        lst.append(i)

print(lst)
lst2=list()
for i in lst:
    m = 1
    n = i
    if n >= 65:

        dd=chr(n) * m
        lst2.append(dd)
for i in lst2:
    print(i, end="")
