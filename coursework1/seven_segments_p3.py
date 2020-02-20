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
    N = X.shape[1] # Number of Dimensions
    K = X.shape[0] # Number of Patterns
    print("W_Pattern_Shape is : ", N, K)

    W = np.zeros([N,N])
    print("Initial Zeros matrix: " , W)
    for i in range(N):
        for j in range(N):
            if i == j:
                W[i,j] = 0
            else:
                for m in range(K):
                    W[i,j] += X[m,i] * X[m,j]
                W[j,i] = W[i,j]  
    W = W/N
    return W

def update_synch(weight,vector,threshold,method):
    # print("Vector type is :",type(vector))
    time_step = 0
    times = 4
    #Print data:
    seven_segment(vector)
    submission.seven_segment(vector)
    Test_1_Energy = energy(weight,vector)
    submission.print_number(Test_1_Energy)
    #Run over 
    if method == "RunAll":
        for update_times in range(times):
            time_step= time_step+1
            current_vector = vector
            for update_index in range(len(vector)):
                next_value = np.dot(weight[update_index][:],vector) - threshold
                if next_value >= 0 :
                    vector[update_index] = 1
                if next_value < 0:
                    vector[update_index] = -1 
            if current_vector == vector:
                break
    #Run step by step
    if method == "RunStep":
        for update_times in range(times):
            time_step= time_step+1
            current_vector = vector
            for update_index in range(len(vector)):
                next_value = np.dot(weight[update_index][:],vector) - threshold
                if next_value >= 0 :
                    vector[update_index] = 1
                if next_value < 0:
                    vector[update_index] = -1 
            # submission.section("The Step  " + str(update_times+1) + " network")
            # submission.matrix_print("Test Vector",[vector])
            seven_segment(vector)
            submission.seven_segment(vector)
            Test_1_Energy = energy(weight,vector)
            submission.print_number(Test_1_Energy)
            if current_vector == vector:
                return vector,time_step       
    return vector,time_step
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
    Associate = np.array([six,three,one])
    print("Associate the patterns is: ", Associate)
    weight_matrix = create_W_pattern(Associate)
    print("Store Patterns is: ", weight_matrix)
    ##this assumes you have called your weight matrix "weight_matrix"
    submission.section("Weight matrix")
    submission.matrix_print("W",weight_matrix)

    print("test1")
    submission.section("Test 1")

    test1=[1,-1,1,1,-1,1,1,-1,-1,-1,-1]
    # Updata the net
    Test_result_1 , Step_1 = update_synch(weight_matrix,test1,0,"RunStep")
    # Output
    # print("Test_result_1: ", Test_result_1)
    # print("Test_1_Seven_segment show： ")
    seven_segment(Test_result_1)
    # submission.seven_segment(Test_result_1)
    # for COMSM0027

    # where energy is the energy of test
    # Test_1_Energy = energy(weight_matrix,test1)
    # submission.print_number(Test_1_Energy)

    ##this prints a space
    submission.qquad()
    # Test_result_1 , Step_1 = update_synch(weight_matrix,test1,0,"RunStep")
    #here the network should run printing at each step
    #for the final submission it should also output to submission on each step

    print("test2")

    test2=[1,1,1,1,1,1,1,-1,-1,-1,-1]
    submission.section("Test 2")
    # # Updata the net
    Test_result_2 , Step_2= update_synch(weight_matrix,test2,0,"RunStep")
    # print("Test_result_2: ", Test_result_2)
    # print("Test_2_Seven_segment show： ")
    # print("Test_2_Steps is : ", Step_2)
    # # Output
    seven_segment(Test_result_2)
    # submission.seven_segment(Test_result_2)
    
    # ##for COMSM0027
    # ##where energy is the energy of test
    # Test_2_Energy = energy(weight_matrix,test2)
    # submission.print_number(Test_2_Energy)

    # ##this prints a space
    # submission.qquad()
    # Test_result_2 , Step_2= update_synch(weight_matrix,test2,0,"RunStep")
    # #here the network should run printing at each step
    # #for the final submission it should also output to submission on each step


    submission.bottomer()



