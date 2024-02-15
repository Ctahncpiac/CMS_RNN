class SimpleAlgo:
    def __init__(self, cluster_simulator):
        self.cluster_simulator = cluster_simulator

    def is_signal(self, cluster):
        # Check if the cluster is a signal based on some criteria
        # define your criteria here
        total_charge = sum(cluster)
        if total_charge > 10:  # Define your threshold for signal detection
            return True
        else:
            return False

    def evaluate_performance(self, num_samples=1000):
        # Initialize counters
        signal_detected = 0
        background_detected = 0
        false_positives = 0

        # Simulate clusters and evaluate performance
        for _ in range(num_samples):
            cluster = self.cluster_simulator.generate_MIP_cluster()
            if self.is_signal(cluster):
                signal_detected += 1
            else:
                background_detected += 1

            # Check false positives
            if sum(cluster) > 10:  # Consider it as a signal
                if not self.is_signal(cluster):
                    false_positives += 1

        # Calculate rates
        signal_efficiency = signal_detected / num_samples
        background_efficiency = background_detected / num_samples
        false_positive_rate = false_positives / num_samples

        return signal_efficiency, background_efficiency, false_positive_rate


# Example of using the SimpleAlgo class
if __name__ == "__main__":
    simulator = ClusterSimulator("config1.json")
    algo = SimpleAlgo(simulator)
    signal_efficiency, background_efficiency, false_positive_rate = algo.evaluate_performance()
    print("Signal Efficiency:", signal_efficiency)
    print("Background Efficiency:", background_efficiency)
    print("False Positive Rate:", false_positive_rate)
