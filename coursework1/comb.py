from math import *
import numpy as np

def seven_segment(pattern):
    def to_bool(a):
        if a == 1:
            return True
        return False

    def hor(d):
        if d:
            print(" _ ")
        else:
            print("   ")

    def vert(d1, d2, d3):
        word = ""

        if d1:
            word = "|"
        else:
            word = " "

        if d3:
            word += "_"
        else:
            word += " "

        if d2:
            word += "|"
        else:
            word += " "

        print(word)

    pattern_b = list(map(to_bool, pattern))

    hor(pattern_b[0])
    vert(pattern_b[1], pattern_b[2], pattern_b[3])
    vert(pattern_b[4], pattern_b[5], pattern_b[6])

    number = 0
    for i in range(0, 4):
        if pattern_b[7 + i]:
            number += pow(2, i)
    print(int(number))

six = [1, 1, -1, 1, 1, 1, 1, -1, 1, 1, -1]
three = [1, -1, 1, 1, -1, 1, 1, 1, 1, -1, -1]
one = [-1, -1, 1, -1, -1, 1, -1, 1, -1, -1, -1]

test1 = [1, -1, 1, 1, -1, 1, 1, -1, -1, -1, -1]

#seven_segment(test)

# here the network should run printing at each step

test2 = [1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1]

#seven_segment(test)

# here the network should run printing at each step
#Create a weight matrix of wijs.

def createweight(x_vec):
    N = len(x_vec)
    w = np.zeros([N,N])
    for i in range(N):
        for j in range(i,N):
            if i==j:
                w[i,j] = 0
            else:
                w[i,j] = x_vec[i]*x_vec[j]
                w[j,i] = w[i,j]
    return w

#MP formula
def update(test,Weight):
    test_ = np.zeros_like(test)
    N = len(test)
    m = 0
    for i in range(N):
            m = test[i]+np.dot(Weight[i][:],test)
            if m > 0:
                test_[i]=1
            else:
                test_[i]=-1
    #print(test_)
    return test_

def energy(test,Weight):
    N = len(test)
    E = 0
    for i in range(N):
       for j in range(N):
           E = E-test[i]*Weight[i,j]*test[j]/2.0
    print(E)
    return E

Weight = (createweight(one) + createweight(three) + createweight(six)) / 3.0
#print out the energy of the three learned patterns
print("the energy of the three learned patterns")
energy(test=one,Weight=Weight)
energy(test=six,Weight=Weight)
energy(test=three,Weight=Weight)

print("test1")
flag =0
while(flag <1):
    a=energy(test=test1,Weight=Weight)
    b=update(test=test1,Weight=Weight)
    c=energy(test=b,Weight=Weight)
    if a-c==0:
        print(b)
        flag=0
    else:
        flag=2
seven_segment(b)

print("test2")
flag_=0
while (flag_ < 1):
    a_=energy(test=test2,Weight=Weight)
    b_=update(test=test2,Weight=Weight)
    c_=energy(test=b_,Weight=Weight)
    if a_-c_==0:
        print(b_)
        flag_=0
    else:
        flag_=2
seven_segment(b_)
