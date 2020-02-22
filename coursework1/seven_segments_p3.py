#for the submission uncomment the submission statements
#so submission.README

from math import *
from submission import *
import numpy as np
import time
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

def create_W_pattern(X):
    N = len(X) # Number of Dimensions
    print("W_Pattern_Shape is : ", N)
    # print(X)
    W = np.zeros([N,N])
    print("Initial Zeros matrix: " , W)
    for i in range(N):
        for j in range(N):
            if i == j:
                W[i,j] = 0.00
            else:
                W[i,j] = X[i] * X[j]
                W[j,i] = W[i,j]
    return W

def update_synch(weight,vector,threshold):
    vector_ = vector
    for update_index in range(len(vector)):
        # next_value = np.dot(weight[update_index][:],np.array(vector)) - threshold
        next_value = 0
        for j in range(11):
            if update_index!=j:
                # print("VECTOR:j is ",vector[j])
                # print("WEIGHT:j is ",weight[update_index,j])
                next_value = next_value + ((vector[j] * weight[update_index,j]))
        # print("VALUE: ", next_value)
        # if next_value >= 0 :
        #     vector_[update_index] = 1
        # else:
        #     vector_[update_index] = -1    
        if np.isclose(next_value,0):
            next_value = 0
        if next_value > 0 :
            vector_[update_index] = 1
        else:
            vector_[update_index] = -1            
    return vector_
def energy(weight,x):
    x = np.array(x)
    Energy = -(x.dot(weight).dot(x.T))/2
    
    return Energy
# The main function
if __name__ == '__main__':
    submission=Submission("Yuli Zhi")
    submission.header("Yuli Zhi")

    six=  [1,1,-1,1,1,1,1,-1,1,1,-1] #0110
    three=[1,-1,1,1,-1,1,1,1,1,-1,-1] #0011
    one=  [-1,-1,1,-1,-1,1,-1,1,-1,-1,-1] #0001

    # test1=[1,-1,1,1,-1,1,1,-1,-1,-1,-1]
    # test2=[1,1,1,1,1,1,1,-1,-1,-1,-1]
    seven_segment(three)
    seven_segment(six)
    seven_segment(one)

    # Associate the patterns
    weight_matrix = (create_W_pattern(one) + create_W_pattern(three) + create_W_pattern(six))/3.0
    print("Store Patterns is: ", weight_matrix)
    ##this assumes you have called your weight matrix "weight_matrix"
    submission.section("Weight matrix")
    submission.matrix_print("W",weight_matrix)

    print("test1")
    submission.section("Test 1")

    test1=[1, -1, 1, 1, -1, 1, 1 , -1, -1, -1, -1]
    print("Test1 vector ", test1)
    test_tmp1 = test1
    # Updata the net
    test_tmp1 = np.zeros_like(test1)
    print(test_tmp1)
    for times in range(10):
        print("The number 's iterate: ", times)
        seven_segment(test1)
        submission.seven_segment(test1)
        # where energy is the energy of test
        Test_1_Energy = energy(weight_matrix,test1)
        submission.print_number(Test_1_Energy)
        # 
        test1 = update_synch(weight_matrix,test1,0)
        if (np.array(test_tmp1)==np.array(test1)).all():
            break
        test_tmp1 = test1
    seven_segment(test1)
    submission.seven_segment(test1)
    Test_1_Energy = energy(weight_matrix,test1)
    submission.print_number(Test_1_Energy)
    submission.qquad()
    # submission.seven_segment(test1)
    # for COMSM0027



    ##this prints a space
    submission.qquad()
    # Test_result_1 , Step_1 = update_synch(weight_matrix,test1,0,"RunStep")
    #here the network should run printing at each step
    #for the final submission it should also output to submission on each step

    print("test2")

    test2=[1,1,1,1,1,1,1,-1,-1,-1,-1]
    print("Test2 vector ", test2)
    submission.section("Test 2")
    # # Updata the net
    test_tmp2 = np.zeros_like(test2)
    print(test_tmp2)
    for times in range(3):
        print("The number 's iterate: ", times)
        seven_segment(test2)
        submission.seven_segment(test2)
        Test_2_Energy = energy(weight_matrix,test2)
        submission.print_number(Test_2_Energy)

        test2 = update_synch(weight_matrix,test2,0)
        if (np.array(test_tmp2)==np.array(test2)).all():
            break
        test_tmp2 = test2
    seven_segment(test2)
    submission.seven_segment(test2)
    Test_2_Energy = energy(weight_matrix,test2)
    submission.print_number(Test_2_Energy)
    submission.qquad()
    # for times in range(100):
        # test2 = update_synch(weight_matrix,test2,0)
    # submission.seven_segment(Test_result_2)
    
    # ##for COMSM0027
    # ##where energy is the energy of test


    # ##this prints a space
    # submission.qquad()
    # Test_result_2 , Step_2= update_synch(weight_matrix,test2,0,"RunStep")
    # #here the network should run printing at each step
    # #for the final submission it should also output to submission on each step

    submission.bottomer()



