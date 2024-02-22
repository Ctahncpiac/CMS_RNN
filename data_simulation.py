import time
from ROOT import TRandom3
import math as m
import numpy as np
import random
import matplotlib.pyplot as plt
import json
from Trajectory_with_plot.py import ParticleTrajectory


rand = TRandom3(int(time.time()))
class ClusterSimulator:
    def __init__(self, config_file):
        self.config_file = config_file
        self.theta = None  # Initialisation de theta à None
        self.pos = None
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
    
    
    def distr_theta(self):
        # Générer une variable aléatoire uniforme entre 0 et 1
        u = np.random.rand()

        # Calculer l'angle theta en fonction de la distribution x^2
        theta = np.arcsin(np.sqrt(u) * (np.sin(np.pi/3)))

        # Choisir aléatoirement la direction de l'angle
        if np.random.rand() < 0.5:
        theta = -theta
        
        return theta

    
    def generate_MIP_cluster(self):
        self.pos = self.generate_position()
        self.theta = self.distr_theta()  # Maintenant, theta est stocké comme attribut de l'instance
        Q = self.generate_charge(self.generate_mean_charge(self.q, self.sigG), self.sigL)
        Q *= np.cos(self.theta)   # Utilisation de self.theta ici
        
        
        # Create an array of size 10 for the cluster
        clus = [0] * 10
        
        # Charge the main charge among 2 clusters based on the input position
        clus[4] = Q * self.pos
        clus[5] = Q * (1 - self.pos)
        
        # Apply a cross-talk effect
        tmp_3 = clus[4] * self.xt1
        tmp_4 = clus[4] * self.xt0 + clus[5] * self.xt1
        tmp_5 = clus[5] * self.xt0 + clus[4] * self.xt1
        tmp_6 = clus[5] * self.xt1
        
        clus[3] = tmp_3
        clus[4] = tmp_4
        clus[5] = tmp_5
        clus[6] = tmp_6
        
        # Add noise
        for i in range(len(clus)):
            clus[i] += rand.Gaus(0, self.noise)
        
        # Apply threshold
        for i in range(len(clus)):
            if clus[i] < 3 * self.noise:
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
