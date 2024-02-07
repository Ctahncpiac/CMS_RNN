import random

class DetectorSignalClassifier:
    def __init__(self, particle_threshold, noise_threshold):
        self.particle_threshold = particle_threshold
        self.noise_threshold = noise_threshold

    def classify_signal(self, signal_metrics):
        if signal_metrics >= self.particle_threshold:
            return "Particle Signal"
        elif signal_metrics <= self.noise_threshold:
            return "Background Noise"
        else:
            return "Uncertain"

def generate_particle_signal():
    # Generate random signal metrics for particle signals
    return random.uniform(800, 1200)

def generate_background_noise():
    # Generate random signal metrics for background noise
    return random.uniform(0, 50)

if __name__ == "__main__":
    # Example thresholds (you need to adjust these based on your specific detector and data)
    particle_threshold = 1000
    noise_threshold = 50

    # Create an instance of the signal classifier
    classifier = DetectorSignalClassifier(particle_threshold, noise_threshold)

    # Generate dataset
    num_samples = 1000
    dataset = []

    for _ in range(num_samples):
        if random.random() < 0.5:  # 50% chance of generating a particle signal
            signal_metrics = generate_particle_signal()
            label = "Particle Signal"
        else:
            signal_metrics = generate_background_noise()
            label = "Background Noise"

        # Classify the signal
        classification = classifier.classify_signal(signal_metrics)

        # Append sample to dataset
        dataset.append((signal_metrics, label, classification))

    # Print the dataset
    for sample in dataset[:10]:  # Print the first 10 samples
        print("Signal Metrics:", sample[0], "| True Label:", sample[1], "| Predicted Label:", sample[2])
