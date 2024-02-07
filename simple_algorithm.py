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

if __name__ == "__main__":
    # Example thresholds (you need to adjust these based on your specific detector and data)
    particle_threshold = 1000
    noise_threshold = 50

    # Create an instance of the signal classifier
    classifier = DetectorSignalClassifier(particle_threshold, noise_threshold)

    # Example signal metrics (you need to replace this with actual signal metrics)
    signal_metrics = 800

    # Classify the signal
    classification = classifier.classify_signal(signal_metrics)

    # Print the classification result
    print("Signal Classification:", classification)

