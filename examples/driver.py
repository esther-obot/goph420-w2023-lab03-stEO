import numpy as np
from lab03.regression import multi_regress
import matplotlib.pyplot as plt

def main(): 
    data=np.loadtxt("examples/M_data.txt")
    # taking data set from M_data.txt 
    t_data = data[:,0]
    #the scale of t axis
    M_data = data[:,1]
    #the scale of M_data axis
    M= np.linspace(-0.5,1,10)
    #the scale andstarting values for M_data
    N= np.zeros_like(M)
    #N array
    for k, Mk in enumerate(M): 
        N[k] = np.sum(np.where(M_data>=Mk,1,0))
    y= np.log10(N)
    #implementation of the linearized formula
    Z= np.vstack([np.ones_like(M),-M]).T
    a,e,rsq=multi_regress(y,Z)
    t_break =np.array([0,35,45,72,95,120])
    #intervals for the M vs N graph

   
   
   
   
   
   
    plt.figure(figsize=(8,10))
    #the graph size implemetation


    plt.subplot(2,1,1)
    plt.plot(t_data,M_data,"ok",markersize=1.0)
    for tb in t_break:
        plt.plot([tb,tb],[-1.5,1.5],"--r")
    plt.xlabel("Time hrs")
    plt.ylabel("M")  
    #plotting for the M vs time graph
   
   
    plt.subplot(2,1,2)
    plt.semilogy(M,N,"ok") 
    plt.semilogy(M,10**(Z@a),"--r")
    plt.title (f"a={a[0]:0.4f},b={a[1]:0.4f},rsq={rsq:0.4f}")
    plt.xlabel("M")
    plt.ylabel("N")
    plt.savefig("examples/all_data.png")
    
    plt.figure(figsize= (8,10))
    for j in range(len(t_break)-1):
        for k, Mk in enumerate(M): 
            N[k] = np.sum(np.where((M_data>=Mk) &  (t_data>=t_break[j])& (t_data<=t_break[j+1]),1,0))
        y= np.log10(N)
        Z= np.vstack([np.ones_like(M),-M]).T
        a,e,rsq=multi_regress(y,Z)

        plt.subplot(3,2,j+1)
        plt.semilogy(M,N,"ok") 
        plt.semilogy(M,10**(Z@a),"--r")
        plt.title (f"a={a[0]:0.4f},b={a[1]:0.4f},rsq={rsq:0.4f}")
        plt.ylim((1,1e4))
        plt.text(-0.5,5,f"{t_break[j]}<=t<={t_break[j+1]}")
        plt.xlabel("M")
        plt.ylabel("N")
        #plotting for the M vs N graph with time interval stamps
    plt.savefig("examples/period stamps.png")




if __name__ == "__main__":
    main()