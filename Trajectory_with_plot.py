import numpy as np
import matplotlib.pyplot as plt
import random as rand

class ParticleTrajectory:
    def generate_position(self,D):
        self.D = D
        return rand.uniform(0, self.D)  # retourne un réel compris entre 0 et D (de manière uniforme)
    
    def distr_theta(self):                                              
        theta = np.random.uniform(np.pi / 4, np.pi / 2)
        if np.random.rand() < 0.5:
            return np.pi / 2 - theta  # Particule partant vers la droite
        else:
            return -(np.pi / 2 - theta)  # Particule partant vers la gauche

    def tracer_droites_et_rectangles(self, D, d):
        self.D = D
        self.d = d
        
        plt.figure(figsize=(10, 10))
        plt.plot([0, self.D], [0, 0], 'k')  # Première droite
        plt.plot([0, self.D], [self.d, self.d], 'k')  # Deuxième droite
        
        for i in range(int(self.D) + 1):
            plt.plot([i * self.D / 10, i * self.D / 10], [0, self.d], 'k')  # Droites perpendiculaires
        for i in range(int(self.D)):
            plt.fill([i * self.D / 10, (i + 1) * self.D / 10, (i + 1) * self.D / 10, i * self.D / 10], [0, 0, self.d, self.d], 'gray', alpha=0.3)
        
        y_i = 0
        x_i = self.generate_position(self.D)
        angle = self.distr_theta()
        x_f = x_i + np.tan(angle) * self.d
        y_f = self.d
        if x_f < 0:
            x_f = 0
            y_f = (x_f - x_i) / np.tan(angle)
        elif x_f > self.D:
            x_f = self.D
            y_f = (x_f - x_i) / np.tan(angle)
        
        plt.plot([x_i, x_f], [y_i, y_f], 'r-', linewidth=2)
        plt.xlabel('Distance (unité de longueur)')
        plt.ylabel('Distance (unité de longueur)')
        plt.title('Passage d\'une particule à travers les plans du détecteur')
        plt.legend(['Trajectoire de la particule'])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        self.X = [x_i, x_f]
        self.Y = [y_i, y_f]
        self.calculate_segments(x_i, x_f, angle)

    def calculate_segments(self, x_i, x_f, angle):
        S = [0] * int(self.D)
        d = abs(x_f - x_i)
        
        if x_f == self.D:  # to ensure the case where x_f=D and int(x_i)=D-1
            x_f = self.D - 1 

        d_int = int(x_f) - int(x_i)

        if d_int > 0: # theta > 0
            S[int(x_i)] = int(x_i + 1) - x_i
            d = d - S[int(x_i)]

            j = 1
            while d > 1:
                S[int(x_i) + j] = 1.0
                d -= 1
                j += 1

            S[int(x_f)] = d
        elif d_int < 0:  # theta < 0
            S[int(x_i)] = x_i - int(x_i)
            d = d - S[int(x_i)]

            j = 1
            while d > 1:
                S[int(x_i) - j] = 1.0
                d -= 1
                j += 1

            S[int(x_f)] = d
        else:
            S[int(x_i)] = d  # case where x_i and x_f are in the same column

        self.S = S
        if angle != 0:
            self.C = [i / np.sin(abs(angle)) for i in S]
        else:
            self.C = [0] * len(S)  # Avoid division by zero

    def ChargeCluster(self, cross_talk_factor=0.5):
        K = 1.5  # Appliquer le facteur de gain initial
        C_modified = [charge * K for charge in self.C]  # Applique le facteur K à chaque charge

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
D = 10 # Largeur de l'espace entre les deux plans
d = 3.0  # Hauteur de chaque petit rectangle
particle_trajectory.tracer_droites_et_rectangles(D, d)
X, S, C = particle_trajectory.X, particle_trajectory.S, particle_trajectory.C
print("Coordonnées initiale - finale :", X)
print("Projection sur l'abscisse :", S)
print("Segmentation", C)
print("Distance totale parcourue :", sum(C))
Charge_déposée = particle_trajectory.ChargeCluster(cross_talk_factor=0.1)  # Ajustez le facteur selon le besoin
print("Charge déposée avec cross-talk :", Charge_déposée)
