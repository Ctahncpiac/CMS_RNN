import time
from ROOT import TRandom3
import math as m
import numpy as np
import random
import matplotlib.pyplot as plt
import json



rand = TRandom3(int(time.time()))

class ClusterSimulator:

    def __init__(self, config_file):
        self.config_file = config_file
        self.load_config(config_file)
      
    
    def load_config(self, config_file):
        with open(config_file) as f:
            config = json.load(f)
        self.r = config["r"]
        self.t = config["t"]
        self.w = config["w"]
        self.noise = config["noise"] 
        
        
    def generate_position(self,w):
        self.w=w
        return random.uniform(0,self.w)  #retoune un réel compris entre 0 et r (de manière uniforme)
    
    def distr_theta(self):

        x = rand.Gaus(np.pi/9, 0.2)
        x = x * random.choice([-1, 1])
        return x
    
    def segmentation(self, r, t ,w):
        self.r=r
        self.t=t
        self.w=w
        #Calculs of initial and final positions

        #real coordinates
        y_i=0
        x_i = self.generate_position(self.w)
        theta = self.distr_theta()
        x_f=x_i+np.tan(theta)*self.t
        y_f=self.t

        if x_f < 0 :
            x_f=0
            y_f=(x_f-x_i)/np.tan(theta)

        elif x_f > self.w:

            x_f=self.w
            y_f=(x_f-x_i)/np.tan(theta)
        
         
        X=[x_i,x_f]
        Y=[y_i,y_f]

        #rescale coordinates 
        x_i= x_i*(self.r/self.w)
        x_f= x_f*(self.r/self.w) 

        #Attribution segment 
        S=[0]*self.r
        d=abs(x_f-x_i)

        if x_f == self.r:  #to ensure the case where x_f=D and int(x_i)=D-1 which will lead to d_int=1
            x_f = self.r -1 

        d_int= int(x_f)-int(x_i)

        if d_int > 0: #theta > 0

            S[int(x_i)]=int(x_i+1)-x_i 
            d=d-S[int(x_i)]

            j=1    
            while d>1:
                S[int(x_i)+j]=1.0
                d-=1
                j+=1

            S[int(x_f)]=d    
              
        elif d_int < 0:  #theta < 0

            S[int(x_i)]= x_i - int(x_i)
            d=d-S[int(x_i)]

            j=1
            while d>1:
                S[int(x_i)-j]=1.0
                d-=1
                j+=1

            S[int(x_f)]=d 
        else :
            S[int(x_i)]=d     #case where x_i and x_f are in the same column

       
        K = 30  # Appliquer le facteur de gain initial

        if theta !=0 :

            C = [ K * (i*self.w)/(self.r * np.sin(abs(theta))) for i in S]
        else: 

            C=[0]*self.r
            C[int(x_i)]=self.t*K 


        return [X,Y,S,C]


##########################################################################################
#                        Functions that will generate MIP and 2MIP                       #
##########################################################################################

    def ct_noise(self,clus_ct):
        
        self.clus_ct = clus_ct  # Ajustez le facteur selon le besoin
       
        #Add cross-talk effect        
        tx0=0.1  #coeff  neighbourg
        tx1 = 0.8 #self coeff

        clus_ct = [0] * len(self.clus_ct)
        clus_ct[0]=self.clus_ct[0]*tx1+self.clus_ct[1]*tx0
        clus_ct[-1]=self.clus_ct[-1]*tx1+self.clus_ct[-2]*tx0

        for i in range(1,len(self.clus_ct)-1):
            clus_ct[i]=self.clus_ct[i]*tx1+self.clus_ct[i-1]*tx0+self.clus_ct[i+1]*tx0
        # Add noise
    
        for i in range(len(clus_ct)):
            clus_ct[i] += rand.Gaus(0, self.noise)
        
        # Apply threshold
        for i in range(len(clus_ct)):
            if clus_ct[i] < 3 * self.noise: #the 3 value is arbitrary
                clus_ct[i] = 0
        return clus_ct
    

    def generate_MIP_cluster(self):

        return self.ct_noise(self.segmentation(self.r,self.t,self.w)[3])
    
    def generate_2MIP_cluster(self):

        clus1 = self.segmentation(self.r,self.t,self.w)[3]
        clus2 = self.segmentation(self.r,self.t,self.w)[3]

        clus3=[0]*len(clus1)
        for i in range(len(clus1)):
            clus3[i]=clus1[i]+clus2[i]       #add up clus before cross talk and noise

        return self.ct_noise(clus3)

    
    def set_config_file(self, config_file):
        self.config_file = config_file
        self.load_config(config_file)


# Example of using the ClusterSimulator class
if __name__ == "__main__":
    simulator = ClusterSimulator("config1.json")
    print(simulator.q) # Example of access to a configured variable
    print("Angle theta généré aléatoirement (en degrés) :", simulator.distr_theta()* (180/np.pi))
    liste1 = simulator.generate_MIP_cluster()
    liste2 = simulator.generate_2MIP_cluster()
    print(liste1)
    print(liste2)

   # Load a new configuration file and compare results
    simulator.set_config_file("config2.json")
    print(simulator.q)  # New value for variable q
    liste3 = simulator.generate_MIP_cluster()
    liste4 = simulator.generate_2MIP_cluster()
    print(liste3)
    print(liste4)
