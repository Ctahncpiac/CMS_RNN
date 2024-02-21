import numpy as np
import matplotlib.pyplot as plt

# Paramètres du détecteur
D = 4  # Longueur des pistes
d = 1   # Largeur des pistes
nombre_pistes = 20

# Générer un nouvel angle d'incidence aléatoire entre π/4 et π/2
angle_incidence = np.random.uniform(np.pi / 4, np.pi / 2)

# Convertir l'angle d'incidence de radians en degrés pour l'affichage
angle_degrees = np.degrees(angle_incidence)

# Générer la position initiale de la particule sur l'axe des abscisses de manière aléatoire entre 0 et 10d
position_initiale_x = np.random.uniform(0, nombre_pistes * d)

# Créer la figure et les axes pour la simulation
plt.figure(figsize=(12, 6))

# Dessiner les pistes
for i in range(nombre_pistes):
    plt.gca().add_patch(plt.Rectangle((i * d, 0), d, D, edgecolor='black', facecolor='none'))

# Initialiser la position courante de la particule
current_position_x = position_initiale_x
current_position_y = 0

# Calculer la trajectoire de la particule
for i in range(nombre_pistes):
    # Assurer un mouvement horizontal minimal pour que la particule traverse les pistes
    min_horizontal_movement = d / 2  # Mouvement horizontal minimal pour assurer la visibilité
    horizontal_movement = max(D / np.tan(angle_incidence), min_horizontal_movement)
    next_position_x = current_position_x + horizontal_movement
    next_position_y = current_position_y + D  # La particule se déplace toujours de D verticalement

    # Tracer la ligne rouge si elle est dans les limites
    if 0 <= next_position_y <= D:
        plt.plot([current_position_x, next_position_x], [current_position_y, next_position_y], 'r-')

    # Mettre à jour la position pour le prochain calcul
    current_position_x = next_position_x
    current_position_y += D  # Avancer verticalement pour la prochaine piste

    # Arrêter si la particule sort du détecteur
    if current_position_x > nombre_pistes * d or current_position_y > D:
        break

# Définir les limites et les titres pour la simulation
plt.xlim(0, nombre_pistes * d)
plt.ylim(0, D)
plt.title("Simulation de la trajectoire d'une particule ")
plt.xlabel('largeur des pistes')
plt.ylabel('épaisseur')
plt.show()





print(angle_degrees)
