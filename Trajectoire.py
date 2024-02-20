import numpy as np
import matplotlib.pyplot as plt
import random as rand

class ParticleTrajectory:
    def generate_position(self):
        # Générer une piste aléatoire entre 0 et 9
        piste = rand.randint(0, 9)
        # Générer une position initiale aléatoire à l'intérieur de cette piste
        return piste * D / 10 + rand.uniform(0, D / 10)
    
    def distr_theta(self):
        # Générer une variable aléatoire uniforme entre 0 et 1
        u = np.random.rand()
        # Calculer l'angle theta en fonction de la distribution x^2
        theta = np.arcsin(np.sqrt(u) * np.sin(np.pi / 3))
        # Choisir aléatoirement la direction de l'angle
        if np.random.rand() < 0.5:
            return np.pi / 2 - theta  # Particule partant vers la droite
        else:
            return -(np.pi / 2 - theta)  # Particule partant vers la gauche

    def tracer_droites_et_rectangles(self, D, d):
        # Initialisations pour le tracé
        plt.figure(figsize=(10, 10))
        plt.plot([0, D], [0, 0], 'k')  # Première droite
        plt.plot([0, D], [d, d], 'k')  # Deuxième droite
        
        # Tracer les droites et rectangles
        for i in range(11):
            plt.plot([i * D/10, i * D/10], [0, d], 'k')  # Droites perpendiculaires
        for i in range(10):
            plt.fill([i * D/10, (i+1) * D/10, (i+1) * D/10, i * D/10], [0, 0, d, d], 'gray', alpha=0.3)

        # Générer des positions et angles aléatoires
        position_initiale = self.generate_position()
        angle_incidence = self.distr_theta()

        # Trajectoire et calculs pour la particule
        traj_x = [position_initiale]
        traj_y = [0]

        # Calcul des distances parcourues dans chaque rectangle
        distances_rectangle = [0] * 10
        i = position_initiale / (D / 10)  # Position initiale relative à la première piste
        N = int(D / (d * np.tan(abs(angle_incidence))))  # Nombre total de pistes traversées

        # Pour la première piste
        distances_rectangle[int(i)] = i * d / np.cos(angle_incidence)

        # Calculs pour les autres pistes jusqu'à la dernière
        for j in range(int(i) + 1, N):
            distances_rectangle[j] = d / np.cos(angle_incidence)

        # Pour la dernière piste, si la particule atteint y = d
        a = D - ((N - 1) - i) * d * np.tan(abs(angle_incidence))
        distances_rectangle[N] = a / np.cos(angle_incidence) if N < 10 else 0  # Assurer que N ne dépasse pas les limites

        # Calculer et tracer la trajectoire réelle de la particule
        # Remarque : Cette partie peut être ajustée si nécessaire pour correspondre à la logique spécifique de votre simulation
        for idx, distance in enumerate(distances_rectangle):
            if distance > 0:  # Si la particule traverse la piste
                x_start = idx * (D / 10)
                x_end = min(x_start + distance * np.sin(angle_incidence), D)
                y_end = min(distance * np.cos(angle_incidence), d)
                traj_x.append(x_end)
                traj_y.append(y_end)

        plt.plot(traj_x, traj_y, 'r-', linewidth=2)

        # Labels et légendes
        plt.xlabel('Distance (unité de longueur)')
        plt.ylabel('Distance (unité de longueur)')
        plt.title('Passage d\'une particule à travers les plans du détecteur')
        plt.legend(['Trajectoire de la particule'])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

        return distances_rectangle

# Utilisation de la classe
particle_trajectory = ParticleTrajectory()
D = 10.0  # Largeur de l'espace entre les deux plans
d = 10.0  # Hauteur de chaque petit rectangle
distances = particle_trajectory.tracer_droites_et_rectangles(D, d)
print("Distances parcourues dans chaque petit rectangle :", distances)
