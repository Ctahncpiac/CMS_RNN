import time
from ROOT import TRandom3
import math as m
import numpy as np
import json


   # def __init__(self):
       # pass

rand = TRandom3(int(time.time()))
class ClusterSimulator:
    def __init__(self, config_file):
        self.config_file = config_file
        self.load_config(config_file)
      
    def load_config(self, config_file):
        with open(config_file) as f:
            config = json.load(f)
        self.q = config["q"]
        self.sigG = config["sigG"]
        self.sigL = config["sigL"]
        self.xt0 = config["xt0"]
        self.xt1 = config["xt1"]
        self.noise = config["noise"] 
        
# Generate a position between 0 and 1 within a strip
    def generate_position(self):
        return rand.Uniform(1)
    
    # Generate the mean charge (mean of Landau)    
    def generate_mean_charge(self, mu, sig):
        return rand.Gaus(mu, sig)
    
    # Generate the total charge    
    def generate_charge(self, mu, sig):
        return rand.Landau(mu, sig)
    
    # Angular distribution for theta in x^2 + 1 between -pi/2 and pi/2
    def distr_theta(self): 
        theta=m.pi/2
        y=2
        while y>m.pow(np.cos(theta),2) + 1 :
            theta = rand.Uniform(-m.pi/2,m.pi/2)  
            y = rand.uniform(0,1)
        return theta

    def generate_MIP_cluster(self):
        pos = self.generate_position()
        Q = self.generate_charge(self.generate_mean_charge(self.q, self.sigG), self.sigL)
        
        # Create an array of size 10 for the cluster
        clus = [0] * 10
        
        # Charge the main charge among 2 clusters based on the input position
        clus[4] = Q * pos
        clus[5] = Q * (1 - pos)
        
        # Apply a cross-talk effect
        tmp_3 = clus[4] * xt1
        tmp_4 = clus[4] * xt0 + clus[5] * xt1
        tmp_5 = clus[5] * xt0 + clus[4] * xt1
        tmp_6 = clus[5] * xt1
        
        clus[3] = tmp_3
        clus[4] = tmp_4
        clus[5] = tmp_5
        clus[6] = tmp_6
        
        # Add noise
        for i in range(len(clus)):
            clus[i] += rand.Gaus(0, noise)
        
        # Apply threshold
        for i in range(len(clus)):
            if clus[i] < 3 * noise:
                clus[i] = 0
        return clus

    def generate_2MIP_cluster(self, delta_pos=3):
        clus1 = self.generate_MIP_cluster()
        clus2 = self.generate_MIP_cluster()
        dp = round(rand.Uniform(1, delta_pos))
        clusd = [0] * 10
        for i in range(len(clusd)):
            if (i + dp) < len(clusd):
                clusd[i] = clus1[i] + clus2[i + dp]
            else:
                clusd[i] = clus1[i]
        return clusd

    def set_config_file(self, config_file):
        self.config_file = config_file
        self.load_config(config_file)


# Example of using the ClusterSimulator class
if __name__ == "__main__":
    simulator = ClusterSimulator("config1.json")
    print(simulator.q) # Example of access to a configured variable

   # Load a new configuration file and compare results
    simulator.set_config_file("config2.json")
    print(simulator.q)  # New value for variable q
