world={'C':7,'D':8}

world['B']=6

print(world)
del world['B']
print(world)

for name in world.keys():
    print(name)

for key,value in world.items():
    print(f"\n Key:{key}")
    print(f"\n Value:{value}")

XuX = {'Yo': 1, 'GO': 5}

Hello=[world,XuX]
for i in Hello:
    print(i)

#rt=XuX.pop('Yo')
#print(XuX,rt)

Go={'rt':1,'Yo':0}
XuX.update(Go)
print(XuX)

L=["App","Bpp","Cpp"]
L[0]="Dpp"

print(L)

K=('Aee','Bee','Cee')

print(K)