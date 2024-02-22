import numpy as np
import matplotlib.pyplot as plt
import random as rand

class ParticleTrajectory:
    def generate_position(self,D):
        self.D=D
        # Générer une piste aléatoire entre 0 et 9
        #piste = rand.randint(0, self.D-1)
        # Générer une position initiale aléatoire à l'intérieur de cette piste
        #return piste * self.D / 10 + rand.uniform(0, self.D / 10)
        return rand.uniform(0,self.D)  #retoune un réel compris entre 0 et D (de manière uniforme)
    
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

    def tracer_droites_et_rectangles(self, D, d):
        self.D=D
        self.d=d
        
        ############
        #Affichage graphique des colonnes
        # Initialisations pour le tracé
        plt.figure(figsize=(10, 10))
        plt.plot([0, self.D], [0, 0], 'k')  # Première droite
        plt.plot([0, self.D], [self.d, self.d], 'k')  # Deuxième droite
        
        # Tracer les droites et rectangles
        for i in range(int(self.D)+1):
            plt.plot([i * self.D/10, i * self.D/10], [0, self.d], 'k')  # Droites perpendiculaires
        for i in range(int(self.D)):
            plt.fill([i * self.D/10, (i+1) * self.D/10, (i+1) * self.D/10, i * self.D/10], [0, 0, self.d, self.d], 'gray', alpha=0.3)
        
        ############



        #Calculs of initial and final positions 
        y_i=0
        x_i = self.generate_position(self.D)
        angle = self.distr_theta()
        x_f=x_i+np.tan(angle)*self.d
        y_f=self.d
        if x_f < 0 :
            x_f=0
            y_f=(x_f-x_i)/np.tan(angle)

        elif x_f > self.D:

            x_f=self.D
            y_f=(x_f-x_i)/np.tan(angle)
        
        X=[x_i,x_f]
        Y=[y_i,y_f]    
        #Attribution segment 
        
        S=[0]*int(self.D)
        d=abs(x_f-x_i)
        
        if x_f == self.D:  #to ensure the case where x_f=D and int(x_i)=D-1 which will lead to d_int=1
            x_f = self.D -1 

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
           

        #Affichage graphique de la trajectoire

        plt.plot(X, Y, 'r-', linewidth=2)

        # Labels et légendes
        plt.xlabel('Distance (unité de longueur)')
        plt.ylabel('Distance (unité de longueur)')
        plt.title('Passage d\'une particule à travers les plans du détecteur')
        plt.legend(['Trajectoire de la particule'])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        C=[0]*self.D
        C[int(x_i)]=self.d

        if angle !=0 :
            C = [ i/np.sin(abs(angle)) for i in S]
         

        return X, S, C
# Utilisation de la classe
particle_trajectory = ParticleTrajectory()
D = 10 # Largeur de l'espace entre les deux plans
d = 10.0  # Hauteur de chaque petit rectangle
X, S , C = particle_trajectory.tracer_droites_et_rectangles(D, d)
print("Coordonées initiale - finale :", X)
print("Projection sur l'abscisse :", S)
print("Segmentation",C)
print("Distance totale parcourue :", sum(C))
