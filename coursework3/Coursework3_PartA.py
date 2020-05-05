#%%
import matplotlib.pyplot as plt
import numpy as np
import math
import random
def Question_1():
    # Question 1
    # for 1s
    # Units
        ms = 0.001
        mv = 0.001
        MO = 1.0e6
        nA = 1.0e-9
    # Param
        tau = 10*ms
        EL = -70*mv
        V_reset = -70*mv
        Vth = -40*mv
        Rm = 10*MO
        Ie = 3.1*nA
    # Euler's method 
        Deta_t = 0.25*ms
        TimeSteps = math.ceil(1/Deta_t)
        print("TimeSteps: ", TimeSteps)
        # Time_Scale = np.arange(0.00000,1.00024,0.25*ms)
        # print(Time_Scale.shape)
        V = []
        V_old = V_reset
        for _ in range(0,TimeSteps):
            V_new = V_old + Deta_t*((EL - V_old + Rm*Ie) / tau)
            if V_new > Vth:
                V_new = V_reset
            V_old = V_new
            V.append(V_new)
        return V,TimeSteps

def Question_2(Es):
    # for 1s
    # Units
        ms = 0.001
        mv = 0.001
        MO = 1.0e6
        nA = 1.0e-9
    # Param
        tau_m = 20*ms
        EL = -70*mv
        V_reset = -80*mv
        Vth = -54*mv
        RmIe = 18*mv
        Rmgs = 0.15
        Deta_s = 0.5
        tau_s = 10*ms
        Deta_t = 0.25*ms
    # Set random initial seed: V_reset and V_threshold
        V1_0 = random.uniform(V_reset,Vth)
        V2_0 = random.uniform(V_reset,Vth)
        V1 = []
        V2 = []
        TimeSteps = math.ceil(1/Deta_t)
        # Es = -0
        # Es = -80*mv
        if Es == 0:
            Type = '_Excitatory'
        if Es == -0.08:
            Type = '_Inhibitory'
        s1 = 0
        s2 = 0
        for _ in range(0,TimeSteps):
            V1_new = V1_0 + Deta_t*((EL - V1_0 + RmIe + Rmgs*(Es - V1_0)*s1) / tau_m )
            V2_new = V2_0 + Deta_t*((EL - V2_0 + RmIe + Rmgs*(Es - V2_0)*s2) / tau_m )
            # v2 = ((1 - (d_t/tau_m)) * v2_old) + (((leak_pot + rmie) * d_t) / tau_m) + (((rmgs * d_t) / tau_m) * s2 * (e_s - v2_old))
            s1 = s1*Deta_t/tau_s
            s2 = s2*Deta_t/tau_s
            if (V1_new > Vth):
                V1_new = V_reset
                s2 += Deta_s
            
            if (V2_new > Vth):
                V2_new = V_reset
                s1 += Deta_s
            V1_0 = V1_new
            V2_0 = V2_new
            V1.append(V1_new)
            V2.append(V2_new)
        plt.figure()
        plt.plot(range(0,TimeSteps),V1,'r',label = 'Neuron1')
        plt.plot(range(0,TimeSteps),V2,'b',label = 'Neuron2')
        plt.plot(range(0,TimeSteps), [Vth]*(TimeSteps),'--', label="Threshold voltage")
        plt.title("Voltage/time  "+Type)
        plt.xlabel("Time/ms")
        plt.ylabel("Voltage/mv")
        plt.legend()
        # plt.show()
        plt.savefig('./coursework3/graphs/'+'Question2'+Type+'.png')

def COMSM2127_2():
    # Use Question1 Para
    # for 1s
    # Question 1
    # for 1s
    # Units
        ms = 0.001
        mv = 0.001
        MO = 1.0e6
        nA = 1.0e-9
    # Param
        tau = 10*ms
        EL = -70*mv
        V_reset = -70*mv
        Vth = -40*mv
        Rm = 10*MO
        Ie = 2.9*nA
    # Euler's method 
        Deta_t = 0.25*ms
        TimeSteps = math.ceil(1/Deta_t)
        print("TimeSteps: ", TimeSteps)
        # Time_Scale = np.arange(0.00000,1.00024,0.25*ms)
        # print(Time_Scale.shape)
        V = []
        # V_old = V_reset
        V_old = -0.05
        for i in range(0,TimeSteps):
            V_new = V_old + Deta_t*((EL - V_old + Rm*Ie) / tau)
            if V_new > Vth:
                # print("Time；", i)
                # print("Voltage: ",V_new)
                V_new = V_reset

            V_old = V_new
            V.append(V_new)
        plt.figure()
        plt.plot(range(0,TimeSteps),V,'g')
        plt.title("Voltage/time with $I_e = 2.9nA$")
        plt.xlabel("Time/ms")
        plt.ylabel("Voltage/mv")
        # plt.show()
        plt.savefig('./coursework3/graphs/COMS2127_2.png')

def COMSM2127_3():
        # Units
    ms = 0.001
    mv = 0.001
    MO = 1.0e6
    nA = 1.0e-9
    # Param
    tau = 10*ms
    EL = -70*mv
    V_reset = -70*mv
    Vth = -40*mv
    Rm = 10*MO
    Ie = 2.9*nA
    I_begin = 2*nA
    I_end = 5*nA
    I_Step = 0.1*nA

    Deta_t = 0.25*ms
    TimeSteps = math.ceil(1/Deta_t)
    Counts = []
    print("TimeSteps: ", TimeSteps)
    for i in range(math.ceil((I_end-I_begin)/I_Step)+1):
        Ie = round(i*0.1+2,2)*nA
        count = 0
        # print(Ie)
        # Euler's method 
        V_old = -0.05
        for i in range(0,TimeSteps):
            V_new = V_old + Deta_t*((EL - V_old + Rm*Ie) / tau)
            if V_new > Vth:
                V_new = V_reset
                count += 1
            V_old = V_new
        Counts.append(count)
    plt.figure()
    plt.plot(np.arange(2,5.1,0.1),Counts,'g')
    plt.title("Number of Spikes with $I_e = 2nA-5nA$")
    plt.xlabel("Ie")
    plt.ylabel("Counts/Spike rate")
    # plt.show()
    plt.savefig('./coursework3/graphs/COMS2127_3.png')
#%%
if __name__ == '__main__':
    # Part A
#%%  Question1
    # V1,T1 = Question_1()
    # plt.plot(range(0,T1),V1,'g')
    # plt.savefig('./coursework3/graphs/Question2.png')
    # plt.show()
#%% Question2
    # Question_2(0)
    # Question_2(-0.08)
#%% COMSM2127
# 1.Compuate the minimial Ie
# Ie = (Vt-EL)/Rm
# Ie = 3nA
# 2. 1nA lower than the minimal Ie
    # COMSM2127_2()
# 3. Calculate Steps:
    # COMSM2127_3()
    print('None')