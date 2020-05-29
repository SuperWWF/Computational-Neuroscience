#%%
import matplotlib.pyplot as plt
import numpy as np
import math
import random
def Possion(DetaT,RInit):
        rand_num = np.random.uniform(0,1,size=[1,40])
        # print(rand_num)
        # rand_num[0,0]=0.003
        rand_num[rand_num>(DetaT*RInit)] = 2
        rand_num[rand_num<(DetaT*RInit)] = 0.5
        rand_num[rand_num>1.5] = 0
        rand_num[rand_num==(DetaT*RInit)] = 0
        # print(rand_num)
        return rand_num

def Question_1():
        # PartB Spike-timing-dependent plasticity
    # Units
        ms = 0.001
        mv = 0.001
        MO = 1.0e6
        nA = 1.0e-9
        nS = 1.0e-9
    # Neuron Param
        tau_m = 10*ms
        EL = -65*mv
        V_reset = -65*mv
        Vth = -50*mv
        Rm = 100*MO
        Ie = 0
        N = 40
    # Synapses
        tau_s = 2*ms
        gi = 4*nS
        Es = 0
        Deta_s = 0.5
        # 40 incoming synapses
        S = np.zeros((1,40)) 
        g = np.full((1,40),gi)
        Deta_t = 0.25*ms
        r_ini = 15 #15 Hz
        # Euler's method 
        print("The g_i is : ",g)
        # Calculate the Time interval
        TimeSteps = math.ceil(1/Deta_t)
        # print("TimeSteps: ", TimeSteps)
        V = []
        V_old = V_reset
        Count = 0
        S0 = S
        for _ in range(0,TimeSteps):
            I_s = Rm*(Es - V_old)*np.sum(S0*gi)
            # print(I_s)
            V_new = V_old + Deta_t*((EL - V_old + Rm*Ie+I_s) / tau_m)
            Possiond = Possion(Deta_t,r_ini)
            # S = S0+S0*Deta_t/tau_s
            # Apply a possion process use random function
            for i in range(0,40):
                rand = random.uniform(0,1)
                if rand < Deta_t*r_ini:
                    S[0,i] = S0[0,i] + 0.5
                else:
                    S[0,i] = S0[0,i] - S0[0,i]*Deta_t/tau_s
            # If there is a spike and update the V value
            if V_new > Vth:
                V_new = V_reset
                Count+=1
            V_old = V_new
            S0 = S
            V.append(V_new)
        print("The fir rate is: ",Count)
        plt.figure()
        plt.plot(range(0,TimeSteps),V,'g')
        plt.title("40 synapses with fixed $\overline {g_i}$ in 1ms")
        plt.xlabel("Time/0.25 ms total 1s wiht 0.25ms intervals")
        plt.ylabel("Voltage/mv")
        plt.savefig('./coursework3/graphs/PartB_Question1.png')
        # plt.show()

def Question_2_essential(STDP_Mode,Inuput_Rate):
    # Question_1()
    # Units
        ms = 0.001
        mv = 0.001
        MO = 1.0e6
        nA = 1.0e-9
        nS = 1.0e-9
    # Neuron Param
        tau_m = 10*ms
        EL = -65*mv
        V_reset = -65*mv
        Vth = -50*mv
        Rm = 100*MO
        Ie = 0
        N = 40
        Recent_Post_Spike = -1000
    # Synapses
        tau_s = 2*ms
        gi = 4*nS
        ga = 2.08*nS
        Es = 0
        Deta_s = 0.5
        # 40 incoming synapses
        S = np.zeros(N)
        # g = np.full(N,gi)
        # g = np.full((1200005,N),0)
        # g[0,:] = gi
        Spikes_Counts = np.zeros(N)
        Recent_Pre_Spikes = np.zeros(N)
        Deta_t = 0.25*ms
        r_ini = Inuput_Rate #15 Hz
    # SDTP par
        A_plus = 0.2*nS
        A_minus = 0.25*nS
        tau_plus = 20*ms
        tau_minus = 20*ms
        # Presynaptic Spike time:
        # t_pre = 0
        # Postsynaptic Spike time:
        # t_post = 0 
        # Set the SDTP flag:
        # STDP_Mode = True
        # STDP_Mode = Flase
        if STDP_Mode:
            g = np.full(N,gi)
        else:
            g = np.full(N,ga)
        # print("The g_i is : ",g)
        # print("The S is : ",S)
        # Calculate the Time interval
        Runtime = 300
        Spike_Average_bin = 10
        TimeSteps = math.ceil(Runtime/Deta_t)
        print("TimeSteps: ", TimeSteps)
        V = []
        V_old = V_reset
        print("The g_i is : ",g)

        # Pre-synaptic neurons N:
        Spike_Count = 0
        Spike_Average = []
        g_average = []
        for t in range(0,TimeSteps):
            I_s = 0
            # Apply a possion process use random function
            # I_s = Rm*(Es - V_old)*np.sum(S*g).all()
            for it in range(0,40):
                I_s += Rm*(Es - V_old)*(S[it]*g[it])
            for i in range(0,40):
                rand = random.uniform(0,1)
                # Spike occures
                if rand < Deta_t*r_ini:
                    S[i] = S[i] + 0.5
                    # print("Pre_Spike")
                    # Spikes_Counts[i] += 1
                    if STDP_Mode:
                        # Store the Pre Spike time
                        Recent_Pre_Spikes[i] = t
                        STDP_Deta_t = (Recent_Post_Spike - Recent_Pre_Spikes[i])
                        # if STDP_Deta_t <= 0:
                        g[i] = g[i] - A_minus * math.exp(-abs((STDP_Deta_t)*Deta_t)/tau_minus)
                        if g[i] < 0:
                            g[i] = 0
                # Spike not occures
                else:
                    S[i] = S[i]-S[i]*Deta_t/tau_s
            # Update The neuron:
            # Post-Synaptic neurons
            # I_s = Rm*(Es - V_old)*np.sum(S*g)
            V_new = V_old + Deta_t*((EL - V_old + Rm*Ie+I_s) / tau_m)
            # If there is a spike and update the V value
            if V_new > Vth:
                V_new = V_reset
                Recent_Post_Spike = t
                Spike_Count +=1 
                # print("Post Spike!!")
                if STDP_Mode:
                    for p in range(0,40):
                        STDP_Deta_t = (Recent_Post_Spike - Recent_Pre_Spikes[i])
                        # if STDP_Deta_t > 0:
                        g[p] = g[p] + A_plus * math.exp(-abs((STDP_Deta_t)*Deta_t)/tau_plus)
                        if g[p] > (4*nS):
                            g[p] = 4*nS
            V_old = V_new
            V.append(V_new)
            if (t+1)%40000 == 0:
                Spike_Average.append(Spike_Count/Spike_Average_bin)
                # print("Spike_Average",Spike_Average)
                Spike_Count = 0
            if t > 800000:
                g_average.append(np.mean(g))
        g_average_all = (np.array(g_average)).mean()
        Spike_Average = np.array(Spike_Average)
        return g,Spike_Average,g_average_all
        # print("Spike_Average is ",Spike_Average)
        # print("g_average: ",g_average)
        # plt.figure()
        # print(g)
        # plt.hist(g, bins=40, facecolor='r', alpha=1.0)
        # plt.xlabel("Weight")
        # plt.ylabel("Frequency")
        # plt.title("Steady-State Synaptic  Distribution")
        # plt.show()

        # plt.plot(range(0,int(Runtime/Spike_Average_bin)),Spike_Average,'g')
        # plt.title("Spike number with 10s bin")
        # plt.xlabel("Time/s 300s total")
        # plt.ylabel("Counts number")
        # # plt.savefig('./coursework3/graphs/PartB_Question1.png')
        # plt.show()

def Question_3_essential(STDP_Mode,Inuput_Rate):
    # Question_1()
    # Units
        ms = 0.001
        mv = 0.001
        MO = 1.0e6
        nA = 1.0e-9
        nS = 1.0e-9
    # Neuron Param
        tau_m = 10*ms
        EL = -65*mv
        V_reset = -65*mv
        Vth = -50*mv
        Rm = 100*MO
        Ie = 0
        N = 40
        Recent_Post_Spike = -1000
    # Synapses
        tau_s = 2*ms
        gi = 4*nS
        ga = 2.08*nS
        Es = 0
        Deta_s = 0.5
        # 40 incoming synapses
        S = np.zeros(N)
        # g = np.full(N,gi)
        # g = np.full((1200005,N),0)
        # g[0,:] = gi
        Spikes_Counts = np.zeros(N)
        Recent_Pre_Spikes = np.zeros(N)
        Deta_t = 0.25*ms
        r_ini = Inuput_Rate #15 Hz
    # SDTP par
        A_plus = 0.2*nS
        A_minus = 0.25*nS
        tau_plus = 20*ms
        tau_minus = 20*ms
        # Presynaptic Spike time:
        # t_pre = 0
        # Postsynaptic Spike time:
        # t_post = 0 
        # Set the SDTP flag:
        # STDP_Mode = True
        # STDP_Mode = Flase
        if STDP_Mode:
            g = np.full(N,gi)
        else:
            g = np.full(N,gi)
        # print("The g_i is : ",g)
        # print("The S is : ",S)
        # Calculate the Time interval
        Runtime = 300
        Spike_Average_bin = 10
        TimeSteps = math.ceil(Runtime/Deta_t)
        print("TimeSteps: ", TimeSteps)
        V = []
        V_old = V_reset
        print("The g_i is : ",g)

        # Pre-synaptic neurons N:
        Spike_Count = 0
        Spike_Average = []
        g_average = []
        for t in range(0,TimeSteps):
            I_s = 0
            # Apply a possion process use random function
            # I_s = Rm*(Es - V_old)*np.sum(S*g).all()
            for it in range(0,40):
                I_s += Rm*(Es - V_old)*(S[it]*g[it])
            for i in range(0,40):
                rand = random.uniform(0,1)
                # Spike occures
                if rand < Deta_t*r_ini:
                    S[i] = S[i] + 0.5
                    # print("Pre_Spike")
                    # Spikes_Counts[i] += 1
                    if STDP_Mode:
                        # Store the Pre Spike time
                        Recent_Pre_Spikes[i] = t
                        STDP_Deta_t = (Recent_Post_Spike - Recent_Pre_Spikes[i])
                        # if STDP_Deta_t <= 0:
                        g[i] = g[i] - A_minus * math.exp(-abs((STDP_Deta_t)*Deta_t)/tau_minus)
                        if g[i] < 0:
                            g[i] = 0
                # Spike not occures
                else:
                    S[i] = S[i]-S[i]*Deta_t/tau_s
            # Update The neuron:
            # Post-Synaptic neurons
            # I_s = Rm*(Es - V_old)*np.sum(S*g)
            V_new = V_old + Deta_t*((EL - V_old + Rm*Ie+I_s) / tau_m)
            # If there is a spike and update the V value
            if V_new > Vth:
                V_new = V_reset
                Recent_Post_Spike = t
                Spike_Count +=1 
                # print("Post Spike!!")
                if STDP_Mode:
                    for p in range(0,40):
                        STDP_Deta_t = (Recent_Post_Spike - Recent_Pre_Spikes[i])
                        # if STDP_Deta_t > 0:
                        g[p] = g[p] + A_plus * math.exp(-abs((STDP_Deta_t)*Deta_t)/tau_plus)
                        if g[p] > (4*nS):
                            g[p] = 4*nS
            V_old = V_new
            V.append(V_new)
            if (t+1)%40000 == 0:
                Spike_Average.append(Spike_Count/Spike_Average_bin)
                # print("Spike_Average",Spike_Average)
                Spike_Count = 0
            if t > 1080000:
                g_average.append(np.mean(g))
        g_average_all = (np.array(g_average)).mean()
        return g,Spike_Average,g_average_all
def Question_4_essential(STDP_Mode,B):
    # Question_1()
    # Units
        ms = 0.001
        mv = 0.001
        MO = 1.0e6
        nA = 1.0e-9
        nS = 1.0e-9
    # Neuron Param
        tau_m = 10*ms
        EL = -65*mv
        V_reset = -65*mv
        Vth = -50*mv
        Rm = 100*MO
        Ie = 0
        N = 40
        Recent_Post_Spike = -1000
    # Synapses
        tau_s = 2*ms
        gi = 4*nS
        ga = 2.08*nS
        Es = 0
        Deta_s = 0.5
        # 40 incoming synapses
        S = np.zeros(N)
        # g = np.full(N,gi)
        # g = np.full((1200005,N),0)
        # g[0,:] = gi
        Spikes_Counts = np.zeros(N)
        Recent_Pre_Spikes = np.zeros(N)
        Deta_t = 0.25*ms
        # r_ini = Inuput_Rate #15 Hz
    # SDTP par
        A_plus = 0.2*nS
        A_minus = 0.25*nS
        tau_plus = 20*ms
        tau_minus = 20*ms
        # Presynaptic Spike time:
        # t_pre = 0
        # Postsynaptic Spike time:
        # t_post = 0 
        # Set the SDTP flag:
        # STDP_Mode = True
        # STDP_Mode = Flase
        if STDP_Mode:
            g = np.full(N,gi)
        else:
            g = np.full(N,ga)
        g_steady = np.zeros(N)
        # print("The g_i is : ",g)
        # print("The S is : ",S)
        # Calculate the Time interval
        Runtime = 300
        Spike_Average_bin = 10
        TimeSteps = math.ceil(Runtime/Deta_t)
        print("TimeSteps: ", TimeSteps)
        V = []
        V_old = V_reset
        print("The g_i is : ",g)
        
        # Question4_Par
        freq = 10 # 10Hz
        r_0 = 20 # 20Hz
        # Pre-synaptic neurons N:
        Spike_Count = 0
        # Spike_Average = []
        # g_average = []
        for t in range(0,TimeSteps):
            I_s = 0
            # <r>(t) = <r>0 + B*sin(2*pai*frequency*t)
            r_ini = r_0 + B * math.sin(2*(math.pi)*freq*((t+1)*Deta_t))
            # Apply a possion process use random function
            # I_s = Rm*(Es - V_old)*np.sum(S*g).all()
            for it in range(0,40):
                I_s += Rm*(Es - V_old)*(S[it]*g[it])
            for i in range(0,40):
                rand = random.uniform(0,1)
                # Spike occures
                if rand < Deta_t*r_ini:
                    S[i] = S[i] + 0.5
                    # print("Pre_Spike")
                    # Spikes_Counts[i] += 1
                    if STDP_Mode:
                        # Store the Pre Spike time
                        Recent_Pre_Spikes[i] = t
                        STDP_Deta_t = (Recent_Post_Spike - Recent_Pre_Spikes[i])
                        if STDP_Deta_t <= 0:
                            g[i] = g[i] - A_minus * math.exp(-abs((STDP_Deta_t)*Deta_t)/tau_minus)
                        else:
                            g[i] = g[i] + A_plus * math.exp(-abs((STDP_Deta_t)*Deta_t)/tau_plus)

                        if g[i] < 0:
                            g[i] = 0
                        if g[i] > (4*nS):
                            g[i] = 4*nS
                # Spike not occures
                else:
                    S[i] = S[i]-S[i]*Deta_t/tau_s
            # Update The neuron:
            # Post-Synaptic neurons
            # I_s = Rm*(Es - V_old)*np.sum(S*g)
            V_new = V_old + Deta_t*((EL - V_old + Rm*Ie+I_s) / tau_m)
            # If there is a spike and update the V value
            if V_new > Vth:
                V_new = V_reset
                Recent_Post_Spike = t
                Spike_Count +=1 
                # print("Post Spike!!")
                if STDP_Mode:
                    for p in range(0,40):
                        STDP_Deta_t = (Recent_Post_Spike - Recent_Pre_Spikes[i])
                        if STDP_Deta_t > 0:
                            g[p] = g[p] + A_plus * math.exp(-abs((STDP_Deta_t)*Deta_t)/tau_plus)
                        else:
                            g[p] = g[p] - A_minus * math.exp(-abs((STDP_Deta_t)*Deta_t)/tau_minus)
                        if g[p] > (4*nS):
                            g[p] = 4*nS
                        if g[p] < 0:
                            g[p] = 0                            
            V_old = V_new
            V.append(V_new)
            if (t+1)%40000 == 0:
                print("Tims is",t/4000)
            #     Spike_Average.append(Spike_Count/Spike_Average_bin)
            #     # print("Spike_Average",Spike_Average)
            #     Spike_Count = 0
            # if t > 1079999:
                # if t == 1080000:
                #     g_steady = np.array(g)
                # else:
                #     g_steady = np.vstack((g_steady,g))
                # g_steady += g
        # g_average_all = g_steady/120000
        return g
def COMSM2127(STDP_Mode):
    # Pre-Setting
        B = 0 # 0Hz
    # Units
        ms = 0.001
        mv = 0.001
        MO = 1.0e6
        nA = 1.0e-9
        nS = 1.0e-9
    # Neuron Param
        tau_m = 10*ms
        EL = -65*mv
        V_reset = -65*mv
        Vth = -50*mv
        Rm = 100*MO
        Ie = 0
        N = 40
        Recent_Post_Spike = 0
    # Synapses
        tau_s = 2*ms
        gi = 4*nS
        ga = 2.08*nS
        Es = 0
        Deta_s = 0.5
        # 40 incoming synapses
        S = np.zeros(N)
        Spikes_Counts = np.zeros(N)
        Recent_Pre_Spikes = np.zeros(N)
        Deta_t = 0.25*ms
        # r_ini = Inuput_Rate #15 Hz
    # SDTP par
        A_plus = 0.2*nS
        A_minus = 0.25*nS
        tau_plus = 20*ms
        tau_minus = 20*ms
        if STDP_Mode:
            g = np.full(N,gi)
        else:
            g = np.full(N,gi)
        # Calculate the Time interval
        Runtime = 300
        TimeSteps = math.ceil(Runtime/Deta_t)
        print("TimeSteps: ", TimeSteps)
        V = []
        V_old = V_reset
        print("The g_i is : ",g)
        # Question4_Par
        freq = 10 # 10Hz
        r_0 = 15 # 20Hz
        # Pre-synaptic neurons N:
        Spike_Count = 0
        # example_pre_index = 17
        Pre_Neuron_Spikes_20s = []
        Pre_Neuron_Spikes_200s = []
        for example in range(0,N,1):
            Pre_Neuron_Spikes_20s.append([])
            Pre_Neuron_Spikes_200s.append([])
        Post_Neuron_Spikes_20s = []
        Post_Neuron_Spikes_200s = []
        for t in range(0,TimeSteps):
            I_s = 0
            # <r>(t) = <r>0 + B*sin(2*pai*frequency*t)
            r_ini = r_0 + B * math.sin(2*(math.pi)*freq*((t+1)*Deta_t))
            # Apply a possion process use random function
            # I_s = Rm*(Es - V_old)*np.sum(S*g).all()
            for it in range(0,40):
                I_s += Rm*(Es - V_old)*(S[it]*g[it])
            for i in range(0,40):
                rand = random.uniform(0,1)
                # Spike occures
                if rand < Deta_t*r_ini:
                    S[i] = S[i] + 0.5
                    # print("Pre_Spike")
                    # Spikes_Counts[i] += 1
                    if t <80000: #first 20 seconds
                        (Pre_Neuron_Spikes_20s[i]).append(t+1)
                    if t >799999: #last 100 seconds
                        (Pre_Neuron_Spikes_200s[i]).append(t+1)
                    if STDP_Mode:
                        # Store the Pre Spike time
                        Recent_Pre_Spikes[i] = t
                        STDP_Deta_t = (Recent_Post_Spike - Recent_Pre_Spikes[i])
                        # if STDP_Deta_t <= 0:
                        g[i] = g[i] - A_minus * math.exp(-abs((STDP_Deta_t)*Deta_t)/tau_minus)
                        if g[i] < 0:
                            g[i] = 0
                # Spike not occures
                else:
                    S[i] = S[i]-S[i]*Deta_t/tau_s
            # Update The neuron:
        # Post-Synaptic neurons
            # I_s = Rm*(Es - V_old)*np.sum(S*g)
            V_new = V_old + Deta_t*((EL - V_old + Rm*Ie+I_s) / tau_m)
            # If there is a spike and update the V value
            if V_new > Vth:
                V_new = V_reset
                Recent_Post_Spike = t
                Spike_Count +=1
                if t < 80000:
                    Post_Neuron_Spikes_20s.append(t+1)
                if t > 799999:
                    Post_Neuron_Spikes_200s.append(t+1)
                # print("Post Spike!!")
                if STDP_Mode:
                    for p in range(0,40):
                        STDP_Deta_t = (Recent_Post_Spike - Recent_Pre_Spikes[i])
                        # if STDP_Deta_t > 0:
                        g[p] = g[p] + A_plus * math.exp(-abs((STDP_Deta_t)*Deta_t)/tau_plus)
                        if g[p] > (4*nS):
                            g[p] = 4*nS
            V_old = V_new
            V.append(V_new)
        
            # x, y = np.random.randn(2, 100)
        diff_20s = []
        diff_200s = []
        for index in range(0,40,1):
            find = Pre_Neuron_Spikes_20s[index]
            for v in find:
                for k in Post_Neuron_Spikes_20s:
                    deta = (v-k)/4.00
                    if deta < 50 or deta>-50: 
                        diff_20s.append(int(deta))
        for index in range(0,40,1):
            find = Pre_Neuron_Spikes_200s[index]
            for v in find:
                for k in Post_Neuron_Spikes_200s:
                    deta = (v-k)/4.00
                    if deta < 50 or deta>-50: 
                        diff_200s.append(int(deta))
        region = []

        for index2 in range(-50,51,1):
            region.append(index2)
        
        hist20s, bins = np.histogram(diff_20s,region)
        hist200s, bins2 = np.histogram(diff_200s,region)
        # for v in Pre_Neuron_Spikes:
        #         for k in Post_Neuron_Spikes:
        #             diff.append(int(v-k))
        print(len(diff_20s))
        print("his: ",(hist20s))
        print("BIns: ",len(bins))
        X = range(-49,51,1)
        title = "Cross-correlogram (STDP: on)"
        plt.figure()
        plt.bar([i - 0.4 for i in X], (hist20s/N)/hist20s[50],width=0.4,color = 'r',label = "Begin 20s")
        plt.bar([i + 0.4 for i in X], (hist200s/N)/hist200s[50],width=0.4,color = 'g',label = "last 100s")
        plt.title(title)
        plt.legend()
        plt.ylabel("coefficient")
        plt.xlabel("time/ms")
        # plt.savefig('q4_cross_corr_off')



        # plt.show()

        # plt.title(title)
        # plt.legend()
        # plt.ylabel("coefficient")
        # plt.xlabel("time/ms")
        # plt.savefig('q4_cross_corr_off')
        plt.show()    
        # Calculate the integration

def COMSM2127_2(STDP_Mode):
    # Pre-Setting
        B = 0 # 0Hz
    # Units
        ms = 0.001
        mv = 0.001
        MO = 1.0e6
        nA = 1.0e-9
        nS = 1.0e-9
    # Neuron Param
        tau_m = 10*ms
        EL = -65*mv
        V_reset = -65*mv
        Vth = -50*mv
        Rm = 100*MO
        Ie = 0
        N = 40
        Recent_Post_Spike = -1000
    # Synapses
        tau_s = 2*ms
        gi = 4*nS
        ga = 2.08*nS
        Es = 0
        Deta_s = 0.5
        # 40 incoming synapses
        S = np.zeros(N)
        Spikes_Counts = np.zeros(N)
        Recent_Pre_Spikes = np.zeros(N)
        Deta_t = 0.25*ms
        # r_ini = Inuput_Rate #15 Hz
    # SDTP par
        A_plus = 0.2*nS
        A_minus = 0.25*nS
        tau_plus = 20*ms
        tau_minus = 20*ms
        if STDP_Mode:
            g = np.full(N,gi)
        else:
            g = np.full(N,gi)
        # Calculate the Time interval
        Runtime = 300
        TimeSteps = math.ceil(Runtime/Deta_t)
        print("TimeSteps: ", TimeSteps)
        V = []
        V_old = V_reset
        print("The g_i is : ",g)
        # Question4_Par
        freq = 10 # 10Hz
        r_0 = 15 # 20Hz
        # Pre-synaptic neurons N:
        Spike_Count = 0
        # example_pre_index = 17
        Pre_Neuron_Spikes_20s = []
        Pre_Neuron_Spikes_200s = []
        for example in range(0,N,1):
            Pre_Neuron_Spikes_20s.append([])
            Pre_Neuron_Spikes_200s.append([])
        Post_Neuron_Spikes_20s = []
        Post_Neuron_Spikes_200s = []
        for t in range(0,TimeSteps):
            I_s = 0
            # <r>(t) = <r>0 + B*sin(2*pai*frequency*t)
            r_ini = r_0 + B * math.sin(2*(math.pi)*freq*((t+1)*Deta_t))
            # Apply a possion process use random function
            # I_s = Rm*(Es - V_old)*np.sum(S*g).all()
            for it in range(0,40):
                I_s += Rm*(Es - V_old)*(S[it]*g[it])
            for i in range(0,40):
                rand = random.uniform(0,1)
                # Spike occures
                if rand < Deta_t*r_ini:
                    S[i] = S[i] + 0.5
                    # print("Pre_Spike")
                    # Spikes_Counts[i] += 1
                    if t <80000: #first 20 seconds
                        (Pre_Neuron_Spikes_20s[i]).append(t/4)
                    if t >799999: #last 100 seconds
                        (Pre_Neuron_Spikes_200s[i]).append(t/4)
                    if STDP_Mode:
                        # Store the Pre Spike time
                        Recent_Pre_Spikes[i] = t
                        STDP_Deta_t = (Recent_Post_Spike - Recent_Pre_Spikes[i])
                        # if STDP_Deta_t <= 0:
                        g[i] = g[i] - A_minus * math.exp(-abs((STDP_Deta_t)*Deta_t)/tau_minus)
                        if g[i] < 0:
                            g[i] = 0
                # Spike not occures
                else:
                    S[i] = S[i]-S[i]*Deta_t/tau_s
            # Update The neuron:
        # Post-Synaptic neurons
            # I_s = Rm*(Es - V_old)*np.sum(S*g)
            V_new = V_old + Deta_t*((EL - V_old + Rm*Ie+I_s) / tau_m)
            # If there is a spike and update the V value
            if V_new > Vth:
                V_new = V_reset
                Recent_Post_Spike = t
                Spike_Count +=1
                if t < 80000:
                    Post_Neuron_Spikes_20s.append(t/4)
                if t > 799999:
                    Post_Neuron_Spikes_200s.append(t/4)
                # print("Post Spike!!")
                if STDP_Mode:
                    for p in range(0,40):
                        STDP_Deta_t = (Recent_Post_Spike - Recent_Pre_Spikes[i])
                        # if STDP_Deta_t > 0:
                        g[p] = g[p] + A_plus * math.exp(-abs((STDP_Deta_t)*Deta_t)/tau_plus)
                        if g[p] > (4*nS):
                            g[p] = 4*nS
            V_old = V_new
            V.append(V_new)
        
            # x, y = np.random.randn(2, 100)
        diff_20s = [0]*101
        diff_200s = [0]*101
        Post_Neuron_Spikes_20s = np.array(Post_Neuron_Spikes_20s)
        Post_Neuron_Spikes_200s =  np.array(Post_Neuron_Spikes_200s)
        X_axis = np.arange(-50,51,1)
        for index in range(0,40,1):
            for Target_Spike20 in Pre_Neuron_Spikes_20s[index]:
                for X_index in range(1,len(X_axis),1):
                    shift20 = Target_Spike20 - Post_Neuron_Spikes_20s
                    count = len(shift20[(shift20>(X_axis[X_index-1])) & (shift20<(X_axis[X_index]))])
                    diff_20s[X_index-1] +=count
        diff_20s = np.array(diff_20s)/N
        # plt.figure()
        # plt.bar(X_axis,diff_20s,width=0.4,color = 'r',label = "Begin 20s")
        # plt.show()

        X_axis = np.arange(-50,51,1)
        for index in range(0,40,1):
            for Target_Spike200 in Pre_Neuron_Spikes_200s[index]:
                for X_index in range(1,len(X_axis),1):
                    shift200 = Target_Spike200 - Post_Neuron_Spikes_200s
                    count = len(shift200[(shift200>(X_axis[X_index-1])) & (shift200<(X_axis[X_index]))])
                    diff_200s[X_index-1] +=count
        diff_200s = np.array(diff_200s)/N
        # plt.figure()
        # plt.bar(X_axis,diff_200s,width=0.4,color = 'r',label = "Last 100s")
        # plt.show()
        return diff_20s,diff_200s
        # plt.bar([i - 0.2 for i in X_axis],diff_20s,width=0.2,color = 'r',label = "Begin 20s")
        # plt.bar([i + 0.2 for i in X_axis],diff_200s,width=0.2,color = 'g',label = "last 100s")
        # plt.title('cross_correlogram')
        # plt.legend()
        # plt.ylabel("coefficient")
        # plt.xlabel("time/ms")
        # plt.show()

        # Calculate the integration
if __name__ == '__main__':
    # Question2_With_STDP_On
    T_A = 3
    g_average = []
    STDP_mean_g = 2.0435e-9
    '''
    for average_t in range(0,T_A):
        print("STDP_ON_Times: ",average_t)
        if average_t == 0:
            G,Spike_Average,g_average_T= Question_2_essential(True,15)
            g_average.append(g_average_T)
        else:
            G_T,Spike_Average_T,g_average_T= Question_2_essential(True,15)
            G = np.vstack((G,G_T))
            Spike_Average = np.vstack((Spike_Average,Spike_Average_T))
            g_average.append(g_average_T)
    print(g_average)
    if T_A == 1:
        g_average_mean = g_average
        Spike_Average_mean = Spike_Average
        g = G
    else:
        g_average_mean = np.mean(g_average)
        g = G.mean(axis = 0)
    # print("STDP_g_average: ",g_average_mean)
    # 
        Spike_Average_mean = Spike_Average.mean(axis = 0)
    Steady_average_fir_rate = np.append(Spike_Average_mean[-3:-1],Spike_Average_mean[-1])
    # print("STDP_Steady_average_fir_rate: ",Steady_average_fir_rate.mean())
    # The result is about 0.5Hz
    # 
    # Plot some figure
    plt.figure()
    plt.hist(g, bins=40, facecolor='r', alpha=1.0)
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.title("Steady-State Synaptic  Distribution")
    # plt.show()
    plt.savefig('./coursework3/graphs/PartB_Question2_STDP_Average_g_1_times.png')

    plt.plot(range(0,int(300/10)),Spike_Average_mean,'g')
    plt.title("Spike rate with 10s bin")
    plt.xlabel("Time/s 300s total")
    plt.ylabel("Counts number")
    plt.savefig('./coursework3/graphs/PartB_Question2_STDP_Average_Fire_Rate_1_times.png')
    # plt.show()
    print (None)

    '''
    
    '''
    
    # Question with SDTP off
    for average_t in range(0,T_A):
        print("SDTP_OFF_Times: ",average_t)
        if average_t == 0:
            G_placeholder,Spike_Average_off,g_average_placeholder= Question_2_essential(False,15)
        else:
            G_placeholder,Spike_Average_T_off,g_average_placeholder= Question_2_essential(False,15)
            Spike_Average_off = np.vstack((Spike_Average_off,Spike_Average_T_off))
    
    Spike_Average_mean_off = Spike_Average_off.mean(axis = 0)
    Steady_average_fir_rate_off = np.append(Spike_Average_mean_off[-3:-1],Spike_Average_mean_off[-1])
    print("STDP_off_Steady_average_fir_rate: ",Steady_average_fir_rate_off.mean())
    
    '''
    '''
    # Question_3
    Spike_rate_SDTP_On  = []
    for rate in range(10,21,1):
        print("Rate is: ", rate)
        for average_t in range(0,T_A):
            if average_t == 0:
                G_placeholder,Spike_Average_on,g_average_placeholder= Question_2_essential(True,rate)
                Spike_Average_on = np.array(Spike_Average_on)
            else:
                G_placeholder,Spike_Average_T_on,g_average_placeholder= Question_2_essential(True,rate)
                Spike_Average_on = np.vstack((Spike_Average_on,Spike_Average_T_on))
        if T_A == 1:
            Spike_Average_mean_on = Spike_Average_on
        else:    
            Spike_Average_mean_on = Spike_Average_on.mean(axis = 0)
        Steady_average_fir_rate_on = np.append(Spike_Average_mean_on[-3:-1],Spike_Average_mean_on[-1])
        Spike_rate_SDTP_On.append(Steady_average_fir_rate_on.mean())
    
    plt.figure()
    plt.plot(range(10,21,1),Spike_rate_SDTP_On,'g')
    plt.title("Steady_fir_rate with SDTP-On")
    plt.xlabel("Input rate/Hz")
    plt.ylabel("Output rate")
    plt.savefig('./coursework3/graphs/PartB_Question3_10_20Hz_SDTP_On.png')
    plt.show()      

    '''
    '''
    Spike_rate_SDTP_Off = []
    for rate in range(10,21,1):
        print("Rate is: ", rate)
        for average_t in range(0,T_A):
            if average_t == 0:
                G_placeholder,Spike_Average_off,g_average_placeholder= Question_2_essential(False,rate)
                Spike_Average_off = np.array(Spike_Average_off)
            else:
                G_placeholder,Spike_Average_T_off,g_average_placeholder= Question_2_essential(False,rate)
                Spike_Average_off = np.vstack((Spike_Average_off,Spike_Average_T_off))
        if T_A == 1:
            Spike_Average_mean_off = Spike_Average_off
        else:
            Spike_Average_mean_off = Spike_Average_off.mean(axis = 0)
        Steady_average_fir_rate_off = np.append(Spike_Average_mean_off[-3:-1],Spike_Average_mean_off[-1])
        Spike_rate_SDTP_Off.append(Steady_average_fir_rate_off.mean())
    
    plt.figure()
    plt.plot(range(10,21,1),Spike_rate_SDTP_Off,'g')
    plt.title("Steady_fir_rate with SDTP-Off")
    plt.xlabel("Input rate/Hz")
    plt.ylabel("Output rate")
    # plt.savefig('./coursework3/graphs/PartB_Question2_STDP_Average_Fire_Rate6_times.png')
    plt.show() 
    '''

    ''' 
    # Run input fir rate 10Hz and 20 Hz SDTP On
    rate_1 = 10
    rate_2 = 20
    for average_t in range(0,T_A):
        if average_t == 0:
            G_rate_1,Spike_Average_on_placeholder,g_average_placeholder= Question_3_essential(True,rate_1)
            G_rate_1 = np.array(G_rate_1)
        else:
            G_rate_1_T,Spike_Average_on_placeholder,g_average_placeholder= Question_3_essential(True,rate_1)
            G_rate_1 = np.vstack((G_rate_1,G_rate_1_T))
    if T_A == 1:
        Gate_1_his_mean = G_rate_1
    else:
        Gate_1_his_mean = G_rate_1.mean(axis = 0)
    plt.figure()
    plt.hist(Gate_1_his_mean, bins=40, facecolor='r', alpha=1.0)
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.title("Steady-State Synaptic Distribution input rate 10Hz")
    plt.show()
    
    for average_t in range(0,T_A):
        if average_t == 0:
            G_rate_2,Spike_Average_on_placeholder,g_average_placeholder= Question_3_essential(True,rate_2)
            G_rate_2 = np.array(G_rate_2)
        else:
            G_rate_2_T,Spike_Average_on_placeholder,g_average_placeholder= Question_3_essential(True,rate_2)
            G_rate_2 = np.vstack((G_rate_2,G_rate_2_T))
    if T_A == 1:
        Gate_2_his_mean = G_rate_2
    else:
        Gate_2_his_mean = G_rate_2.mean(axis = 0)
    plt.figure()
    plt.hist(Gate_2_his_mean, bins=40, facecolor='r', alpha=1.0)
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.title("Steady-State Synaptic Distribution input rate 20Hz")
    plt.show()
    '''
    
    '''
    # Question 4
    B = [0,5,10,15,20]
    g_steady_mean = []
    g_steady_deviation = []
    for B_value in B:
        print("B value is :",B)
        g_steady_t = Question_4_essential(True,B_value)
        g_steady_mean.append(np.mean(g_steady_t))
        g_steady_deviation.append(np.std(g_steady_t))
    
    plt.figure()
    plt.plot(B,g_steady_mean,'g',label = 'Mean')
    plt.plot(B,g_steady_deviation,'r',label = 'Standard Deviation')
    plt.title("The mean and deviation of g_steady")
    plt.ylabel("Mean/Std")
    plt.xlabel("B value")
    plt.legend()
    # plt.savefig('./coursework3/graphs/PartB_Question2_STDP_Average_Fire_Rate6_times.png')
    plt.show()

    '''
    
    
    # Use the B=0/B=20Hz
    B_0_hism = np.zeros(40)
    for average_t in range(0,T_A):
            B_0_hism_T = Question_4_essential(True,0)
            B_0_hism += B_0_hism_T
    if T_A == 1:
        B_0_hism_mean = B_0_hism
    else:
        B_0_hism_mean = B_0_hism/T_A


    plt.figure()
    plt.hist(B_0_hism_mean, bins=40, facecolor='r', alpha=1.0)
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.title("Steady-State Synaptic Distribution with B=0Hz")
    # plt.savefig('./coursework3/graphs/PartB_Questio4_B_0Hz_2.png')
    plt.show()
    
    
    '''
    for average_t in range(0,T_A):
        if average_t == 0:
            B_20_hism = Question_4_essential(True,20)
            B_20_hism = np.array(B_20_hism)
        else:
            B_20_hism_T = Question_4_essential(True,20)
            B_20_hism = np.vstack((B_20_hism,B_20_hism_T))
    if T_A == 1:
        B_20_hism_mean = B_20_hism
    else:
        B_20_hism_mean = B_20_hism.mean(axis = 0)


    plt.figure()
    plt.hist(B_20_hism_mean, bins=40, facecolor='r')
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.title("Steady-State Synaptic Distribution with B=20Hz")
    plt.savefig('./coursework3/graphs/PartB_Questio4_B_20Hz_2.png')
    # plt.show()
    '''
    '''
    # COMSM2127 Question
    # COMSM2127_2(True)

    for average_t in range(0,T_A):
        if average_t == 0:
            Diff_20s,Diff_200s = COMSM2127_2(True)
        else:
            Diff_20s_T,Diff_200s_T = COMSM2127_2(True)
            Diff_20s = np.vstack((Diff_20s,Diff_20s_T))
            Diff_200s = np.vstack((Diff_200s,Diff_200s_T))
    if T_A == 1:
        M_Diff_20s = Diff_20s
        M_Diff_200s = Diff_200s
    else:
        M_Diff_20s = Diff_20s.mean(axis = 0)
        M_Diff_200s = Diff_200s.mean(axis = 0)
    # plt.figure()
    # plt.bar(X_axis,diff_20s,width=0.4,color = 'r',label = "Begin 20s")
    # plt.show()

    # plt.figure()
    # plt.bar(X_axis,diff_200s,width=0.4,color = 'r',label = "Last 100s")
    # plt.show()
    X_axis = np.arange(-50,51,1)
    plt.bar([i - 0.2 for i in X_axis],M_Diff_20s,width=0.2,color = 'r',label = "Begin 20s")
    plt.bar([i + 0.2 for i in X_axis],M_Diff_200s,width=0.2,color = 'g',label = "last 100s")
    plt.title('cross_correlogram')
    plt.legend()
    plt.ylabel("coefficient")
    plt.xlabel("time/ms")
    plt.show()
    '''
# %%
