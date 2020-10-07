#import csv
#w=csv.writer(open("Output.csv","w"))
import json

Details = {}

cond=True

while cond:

    Name = input("Enter the students name")                             #Student's name
    Roll_no = int(input("Enter the roll no. of the student"))           #Students's roll

    Details[Name] = Roll_no                                              #maintaing the data

    json.dump(Details,file('SIH_Output.txt','w'))
    #for Name,Roll_no in Details.items():
        #w.writerow([Name,Roll_no])

    ctr=input("Do you want to continue")
    if ctr == "No" or ctr == "no":
        cond = False

print("\n--- Student Details ---")

for Name , Roll_no in Details.items():                                  #printing the students details
    print(f"\nStudent Name : {Name}")
    print(f"\nStudent Roll No. : {Roll_no}")


