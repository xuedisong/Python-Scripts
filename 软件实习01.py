def check_2(password):
    while password!="Zky":
        password=raw_input("input again:")
    print "success"
        
def check_1(password):
    count=0
    while password!="Zky":
        password=raw_input("input again:")
        count=count+1
        if(count==5):
            print "account locked"
            return
        
    if(password=="Zky"):
        print "success"
        
def GradeJudge(grade):
        if(grade>=90):
            print "A"
        else: 
            if(grade>=80):
                print "B"
            else: 
                if(grade>=70):
                    print "C"
                else: 
                    if(grade>=60):
                        print "D"
                    else:
                        print "E"

originl=[1,1,2,3,4,4,4,7,7,7,7,9,10,10,10]
iarray=[0]
result=list()
originl.sort()
loop=0
index=0
i=0
while(i<len(originl)):
    before=originl[i]
    while originl[i]==before:
        i=i+1
        if i>=len(originl):
            break
    
    ##print i
    iarray.append(i)        
    result.append(originl[i-1])
    index=index+1
    result.append(iarray[index]-iarray[index-1])
    
print result    