#%%
import random as rnd
import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt
def get_spike_train(rate,big_t,tau_ref):

    if 1<=rate*tau_ref:
        print("firing rate not possible given refractory period f/p")
        return []


    exp_rate=rate/(1-tau_ref*rate)
    
    spike_train=[]

    t=rnd.expovariate(exp_rate)
    # print('The Len: ',t)
    while t< big_t:
        spike_train.append(t)
        t+=tau_ref+rnd.expovariate(exp_rate)

    return spike_train

def Fano(Spike_Train,Big_t,Windows):
    # Store the result
    Counts = [0 for i in range(int((round(Big_t/Windows))))]
    Region_index = 0
    for i in range(len(Spike_Train)):
        while True:
            if Spike_Train[i] < (Region_index+1)*Windows:
                Counts[Region_index] +=1
                break
            else:
                Region_index +=1
    return np.var(Counts)/np.mean(Counts)

def Cov(Spike_Train):
    # Store the result
    interval  = [Spike_Train[0]]
    for i in range(len(Spike_Train)):
        if i < (len(Spike_Train)-1):
            interval.append(Spike_Train[i+1]-Spike_Train[i])
    
    return np.std(interval)/np.mean(interval)

def load_data(filename,T):

    data_array = [T(line.strip()) for line in open(filename, 'r')]

    return data_array   

def Fano_Sim(Spike_Train,Windows):
    Counts = []
    # Each data for 2ms
    Index = len(Spike_Train)
    for i in range(0,Index,Windows//2):
        Number = 0
        for j in range(0,Windows//2):
            if Spike_Train[i+j] == 1:
                Number+=1
        Counts.append(Number)
    return np.var(Counts)/np.mean(Counts)
def Cov_Sim(Spike_Train):
    interval = []
    interval_time = 0
    for i in Spike_Train:
        if i == 1:
            interval.append(interval_time)
            interval_time = 0
        interval_time += 0.002
    return np.std(interval)/np.mean(interval)
def Acf(Spike_Train):
    Acf_Y = [0]*101
    for index in range(0,len(Spike_Train)):
        if Spike_Train[index] == 1:
            Acf_Y[50] += 1
            for forward in range(1,51,1):
                if (index+forward) > (len(Spike_Train)-1):
                    break
                if Spike_Train[index+forward] == 1:
                    Acf_Y[50+forward]+=1
            for backward in range(-1,-51,-1):
                if (index+backward) < 0:
                    break
                if Spike_Train[index+backward] == 1:
                    Acf_Y[50+backward]+=1
    return (np.array(Acf_Y)/Acf_Y[50])

def SpikeAverage(Spike_Train,Stimulus):
    interval_counts = 0
    Sum = [0]*50
    for index in range(50,len(Spike_Train)):
        if Spike_Train[index] == 1:
            interval_counts+=1
            StimData = Stimulus[index-50:index-1] # 100ms
            # print(len(StimData))
            for offset in range(0,len(StimData)):
                Sum[offset]+=StimData[offset]
    
    return np.array(Sum)/interval_counts

def MultiSpikeAverage(Spike_Train,Stimulus,interval,adj):
    interval_counts = 0
    Sum = np.zeros(50)
    if adj == 'Flase':
        for index in range(50,len(Spike_Train)):
            if Spike_Train[index] == 1 and Spike_Train[index+interval]:
                interval_counts +=1
                Sum += Stimulus[index-50:index]
        return Sum/interval_counts

    if adj == 'True':
        for index in range(50,len(Spike_Train)):
            if Spike_Train[index] == 1 and Spike_Train[index+interval] and not 1 in Spike_Train[index+1:index+interval]:
                interval_counts +=1
                Sum += Stimulus[index-50:index]
        return Sum/interval_counts
#%%
Hz=1.0
sec=1.0
ms=0.001

rate=35.0 *Hz
tau_ref=5*ms

big_t=1000*sec
# Question1
spike_train_ref =get_spike_train(rate,big_t,tau_ref)
spike_train_no_ref =get_spike_train(rate,big_t,0)
# print(len(spike_train_ref)/big_t)
# print(len(spike_train_no_ref)/big_t)
windows_10ms = 10*ms
windows_50ms = 50*ms
windows_100ms =100*ms
# ref
Fano_10ms_ref = Fano(spike_train_ref,big_t,windows_10ms)
Fano_50ms_ref = Fano(spike_train_ref,big_t,windows_50ms)
Fano_100ms_ref = Fano(spike_train_ref,big_t,windows_100ms)
Cov_ref = Cov(spike_train_ref)
print("Fano_10ms_ref is: ",Fano_10ms_ref)
print("Fano_50ms_ref is: ",Fano_50ms_ref)
print("Fano_100ms_ref is: ",Fano_100ms_ref)
print ("Cov_ref is: ",Cov_ref)
# no ref
Fano_10ms_no_ref = Fano(spike_train_no_ref,big_t,windows_10ms)
Fano_50ms_no_ref = Fano(spike_train_no_ref,big_t,windows_50ms)
Fano_100ms_no_ref = Fano(spike_train_no_ref,big_t,windows_100ms)
Cov_no_ref = Cov(spike_train_no_ref)
print("Fano_10ms_no_ref is: ",Fano_10ms_no_ref)
print("Fano_50ms_no_ref is: ",Fano_50ms_no_ref)
print("Fano_100ms_no_ref is: ",Fano_100ms_no_ref)
print ("Cov_no_ref is: ",Cov_no_ref)
#%%
# Question2
#spikes=[int(x) for x in load_data("rho.dat")]
spikes=load_data("d:/Document/GitHub/Computational-Neuroscience/coursework2/rho.dat",int)
Fano_Sim_10 = Fano_Sim(spikes,10)
Fano_Sim_50 = Fano_Sim(spikes,50)
Fano_Sim_100 = Fano_Sim(spikes,100)
print("Fano_Sim_10 is",Fano_Sim_10)
print("Fano_Sim_50 is",Fano_Sim_50)
print("Fano_Sim_100 is",Fano_Sim_100)
Cov_Sim_1000s = Cov_Sim(spikes)
print("Cov_Sim is: ", Cov_Sim_1000s)
#%%
# Question3
# plot_acf(spikes,lags=50)
X_in = np.arange(-100,102,2)
Y_in = Acf(spikes)
plt.bar(X_in,Y_in,width=1)
plt.xlabel("interval(ms)")
plt.ylabel("AutoCorrelation")
plt.title("Autocorrelogram")
# plot.acorr(np.array(spikes),usevlines=True, normed=True, maxlags=100)
plt.savefig("../graphs/Autocorrelogram.png")
plt.show()
#%%
# Question4
#stimulus=[float(x) for x in load_data("stim.dat")]
#100ms windwos == 50 points
stimulus=load_data("d:/Document/GitHub/Computational-Neuroscience/coursework2/stim.dat",float)
Y_aver = SpikeAverage(spikes,stimulus)
X_aver = np.arange(0,100,2)
plt.plot(X_aver,Y_aver)
plt.xlabel("time(ms)")
plt.ylabel("Average")
plt.title("SpikeAverage")
plt.savefig("../graphs/SpikeAverage.png")
plt.show()
#%%
# Question CMOS
interval_2ms = 1 #2ms
interval_10ms = 5 #10ms
interval_20ms = 10 #20ms
interval_50ms = 25 #50ms
X = np.arange(0,100,2)
Y_2ms_Non = MultiSpikeAverage(spikes,stimulus,interval_2ms, 'True')
Y_2ms_adj = MultiSpikeAverage(spikes,stimulus,interval_2ms, 'Flase')

Y_10ms_Non = MultiSpikeAverage(spikes,stimulus,interval_10ms, 'True')
Y_10ms_adj = MultiSpikeAverage(spikes,stimulus,interval_10ms, 'Flase')

Y_20ms_Non = MultiSpikeAverage(spikes,stimulus,interval_20ms, 'True')
Y_20ms_adj = MultiSpikeAverage(spikes,stimulus,interval_20ms, 'Flase')

Y_50ms_Non = MultiSpikeAverage(spikes,stimulus,interval_50ms, 'True')
Y_50ms_adj = MultiSpikeAverage(spikes,stimulus,interval_50ms, 'Flase')

fig,axs = plt.subplots(2,4,sharex='col',sharey='row')
fig.set_size_inches(18.5, 10.5)
fig.suptitle("Multi_Spikes_Average")
axs[0,0].plot(X,Y_2ms_Non)
axs[0,0].set_title("Interval-2ms-not")

axs[1,0].plot(X,Y_2ms_adj)
axs[1,0].set_title("Interval-2m-adjacent")
############################################################
axs[0,1].plot(X,Y_10ms_Non)
axs[0,1].set_title("10ms-not")

axs[1,1].plot(X,Y_10ms_adj)
axs[1,1].set_title("10ms-adjacent")
###########################################################
axs[0,2].plot(X,Y_20ms_Non)
axs[0,2].set_title("20ms-not")

axs[1,2].plot(X,Y_20ms_adj)
axs[1,2].set_title("20ms-adjacent")
###########################################################
axs[0,3].plot(X,Y_50ms_Non)
axs[0,3].set_title("50ms-not")

axs[1,3].plot(X,Y_50ms_adj)
axs[1,3].set_title("50ms-adjacent")
##########################################################
for ax in axs.flat:
    ax.set(xlabel='Time(ms)', ylabel='SpikeAverage')

for ax in axs.flat:
    ax.label_outer()
fig.savefig("../graphs/MultiAverage.png")
plt.show()






# %%
