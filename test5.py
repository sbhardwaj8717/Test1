for i in range(0,2):
     mul = 1
     n   = i
     if n >= 26:
         n   = n-26
         mul = 2
     print(chr(65+n)*mul)
