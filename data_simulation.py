import time
from ROOT import TRandom3
import numpy as np
import random
import json



rand = TRandom3(int(time.time()))

class ClusterSimulator:

    def __init__(self, config_file):
        self.config_file = config_file
        self.load_config(config_file)

        #Lists that compares who has the highest factor between MIP and 2MIP
            
        self.P=[]
        self.L=[]
      
    
    def load_config(self, config_file):
        with open(config_file) as f:
            config = json.load(f)
        self.b = config["b"]
        self.r = config["r"]
        self.t = config["t"]
        self.w = config["w"]
        self.noise = config["noise"] 
        
        
    def generate_position(self,w):
        self.w=w
        return random.uniform(0,self.w)  #retoune un réel compris entre 0 et r (de manière uniforme)
    
    def distr_theta(self):

        x = rand.Gaus(np.pi/15,0.12)*random.choice([-1, 1])
        return x
    
    def charge(self, r, t ,w,G):

        self.G=G   #Factor : useful to generate the same amount of charge in MIP and 2MIP, in this case for MIP: G=2 and 2MIP: G=1
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

   
        Q=30  #gain for the charge

        if theta !=0 : 

            C = [ Q*self.G*((i*self.w)/(self.r * np.sin(abs(theta)))) for i in S]
        else: 

            C=[0]*self.r
            C[int(x_i)]=self.t*self.G


        return [X,Y,C]
    
    
    def ADC(self,C): #Analog to Digital Converter
        #Add cross-talk effect
        self.C=C        
        tx0 = 0.1  #coeff  neighbourg
        tx1 = 0.8 #self coeff

        if self.r !=1:
            clus_ct = self.C.copy()
            clus_ct[0]=self.C[0]*tx1+self.C[1]*tx0
            clus_ct[-1]=self.C[-1]*tx1+self.C[-2]*tx0
        

            for i in range(1,len(clus_ct)-1):
                clus_ct[i]=self.C[i]*tx1+self.C[i-1]*tx0+self.C[i+1]*tx0
        
        # Add noise
    
        for i in range(len(clus_ct)):
            clus_ct[i] += rand.Gaus(0, self.noise)
        
        # Apply threshold
        for i in range(len(clus_ct)):
            if clus_ct[i] < 2*self.noise: # arbitrary
                clus_ct[i] = 0      

        #ADC  (linear)
        Q=30  #gain for the charge
        q_threshold = 0.85*Q*max(sum(self.P),sum(self.L))*np.sqrt((self.w/self.r)**2+self.t**2) #all value greater than that will be set to 255 

        for i in range(len(clus_ct)):
            if clus_ct[i] >= q_threshold:
                clus_ct[i]=self.b
            else:
                clus_ct[i]=int((clus_ct[i]*self.b)/q_threshold)
        return clus_ct                    
            

        
##########################################################################################
#                        Functions that will generate MIP and 2MIP                       #
##########################################################################################

    

    def generate_MIP_cluster(self,G):
        self.G=G
        
        self.P=[]                   
        self.P.append(self.G)

        clus = self.charge(self.r,self.t,self.w,self.G)[2]

        return self.ADC(clus)
    
    def generate_2MIP_cluster(self,G):

        self.G=G

        self.L=[]
        self.L.append(self.G*2) #*2 since its the sum of 2 MIP

        clus1 = self.charge(self.r,self.t,self.w,self.G)[2]
        clus2 = self.charge(self.r,self.t,self.w,self.G)[2]

        clus3=[clus1[i]+clus2[i] for i in range(len(clus1))]


        return self.ADC(clus3)

    
    def set_config_file(self, config_file):
        self.config_file = config_file
        self.load_config(config_file)


