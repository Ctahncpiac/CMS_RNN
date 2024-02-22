import numpy as np
import matplotlib.pyplot as plt
import random as rand

class ParticleTrajectory:
    def generate_position(self,w):
        self.w=w
        return rand.uniform(0,self.w)  #retoune un réel compris entre 0 et r (de manière uniforme)
    
    def distr_theta(self):
        # Générer une variable aléatoire uniforme entre 0 et 1
        u = np.random.rand()                                                #pq utiliser np.random.rand alors qu'on a rand.uniform(0,1) ?
        # Calculer l'angle theta en fonction de la distribution x^2
        theta = np.arcsin(np.sqrt(u) * np.sin(np.pi / 3))
        # Choisir aléatoirement la direction de l'angle
        if np.random.rand() < 0.5:
            return np.pi / 2 - theta  # Particule partant vers la droite
        else:
            return -(np.pi / 2 - theta)  # Particule partant vers la gauche

    def tracer_droites_et_rectangles(self, r, t ,w):
        self.r=r
        self.t=t
        self.w=w
        #Calculs of initial and final positions

        #real coordinates
        y_i=0
        x_i = self.generate_position(self.w)
        angle = self.distr_theta()
        x_f=x_i+np.tan(angle)*self.t
        y_f=self.t

        if x_f < 0 :
            x_f=0
            y_f=(x_f-x_i)/np.tan(angle)

        elif x_f > self.w:

            x_f=self.w
            y_f=(x_f-x_i)/np.tan(angle)
        
         
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

        C=[0]*self.r
        C[int(x_i)]=self.t

        if angle !=0 :
            C = [ (i*self.w)/(self.r * np.sin(abs(angle))) for i in S]
         

        return X, S, C


    def ChargeCluster(self, cross_talk_factor=0.5):
        K = 1.5  # Appliquer le facteur de gain initial
        C_modified = [charge * K for charge in C]  # Applique le facteur K à chaque charge

        # Appliquer l'effet de cross-talk
        C_with_cross_talk = [0] * len(C_modified)
        for i in range(len(C_modified)):
            C_with_cross_talk[i] += C_modified[i]
            if i > 0:
                C_with_cross_talk[i - 1] += C_modified[i] * cross_talk_factor
            if i < len(C_modified) - 1:
                C_with_cross_talk[i + 1] += C_modified[i] * cross_talk_factor
        return C_with_cross_talk


# Utilisation de la classe
particle_trajectory = ParticleTrajectory()
r = 100 # resolution
t = 14  # epaisseur
w = 17 #largeur
X, S , C = particle_trajectory.tracer_droites_et_rectangles(r,t,w)
print("Coordonées x_i - x_f :", X)
print("Projection sur l'abscisse (unité):", S)
print("Segmentation",C)
print("Distance totale parcourue :", sum(C))
