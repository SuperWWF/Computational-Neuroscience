#for the submission uncomment the submission statements
#so submission.README

from math import *
import numpy as np
from submission import *

def seven_segment(pattern):

    def to_bool(a):
        if a==1:
            return True
        return False
    

    def hor(d):
        if d:
            print(" _ ")
        else:
            print("   ")
    
    def vert(d1,d2,d3):
        word=""

        if d1:
            word="|"
        else:
            word=" "
        
        if d3:
            word+="_"
        else:
            word+=" "
        
        if d2:
            word+="|"
        else:
            word+=" "
        
        print(word)

    

    pattern_b=list(map(to_bool,pattern))

    hor(pattern_b[0])
    vert(pattern_b[1],pattern_b[2],pattern_b[3])
    vert(pattern_b[4],pattern_b[5],pattern_b[6])

    number=0
    for i in range(0,4):
        if pattern_b[7+i]:
            number+=pow(2,i)
    print(int(number))
        
submission=Submission("Jeffrey_Man_Hong_Chu")
submission.header("Jeffrey_Man_Hong_Chu")

six=[1,1,-1,1,1,1,1,-1,1,1,-1]
three=[1,-1,1,1,-1,1,1,1,1,-1,-1]
one=[-1,-1,1,-1,-1,1,-1,1,-1,-1,-1]

seven_segment(three)
seven_segment(six)
seven_segment(one)

##this assumes you have called your weight matrix "weight_matrix"
submission.section("Weight matrix")
def weight_matrix(x):
    N = len(x)
    w = np.zeros([N,N])
    for i in range(N):
        for j in range(i,N):
            if i==j:
                w[i,j] = 0
            else:
                w[i,j] = x[i]*x[j]
                w[j,i] = w[i,j]
    return w

def update(test,weight):
    new_test = np.zeros_like(test)
    N = len(test)
    m = 0
    for i in range(N):
            m = np.dot(weight[i][:],test)
            if m > 0:
                new_test[i]=1
            else:
                new_test[i]=-1
    return new_test

weight = (weight_matrix(one) + weight_matrix(three) + weight_matrix(six)) / 3.0
print(type(weight))
submission.matrix_print("W",weight)

print("test1")

submission.section("Test 1")

def is_equal(list_a,list_b):

    equal=True

    for i in range(0,len(list_a)):
        if list_a[i]!=list_b[i]:
            equal=False

    return equal

test=[1,-1,1,1,-1,1,1,-1,-1,-1,-1]
seven_segment(test)
new_test_copy=[0 for i in range(0,11)]
#while(np.array(test).all()!=np.array(new_test_copy).all()):
submission.seven_segment(test)
while(not is_equal(test,new_test_copy)):
    new_test_copy = new_test_copy
    new_test = update(test,weight)
    test = new_test
    seven_segment(test) 
    submission.seven_segment(test)
##for COMSM0027


##where energy is the energy of test
#submission.print_number(energy)

##this prints a space
submission.qquad()

#here the network should run printing at each step
#for the final submission it should also output to submission on each step


submission.section("Test 2")

print("test2")

test=[1,1,1,1,1,1,1,-1,-1,-1,-1]
new_test_copy=[0 for i in range(0,11)]
seven_segment(test)
#while(np.array(test).all()!=np.array(new_test_copy).all()):
submission.seven_segment(test)
while(not is_equal(test,new_test_copy)):
    new_test_copy = new_test
    new_test = update(test,weight)
    test = new_test
    seven_segment(test)
    submission.seven_segment(test)

##for COMSM0027
##where energy is the energy of test
#submission.print_number(energy)

##this prints a space
submission.qquad()

#here the network should run printing at each step
#for the final submission it should also output to submission on each step


submission.bottomer()