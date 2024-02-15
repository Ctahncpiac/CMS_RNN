from data_simulation import ClusterSimulator

class SimpleAlgo:
    def __init__(self, cluster_simulator):
        self.cluster_simulator = cluster_simulator

    def is_signal(self, cluster, threshold):
        # Check if the cluster is a signal based on the given threshold
        total_charge = sum(cluster)
        if total_charge > threshold:
            return True
        else:
            return False

    def evaluate_performance(self, num_samples=1000):
        # Initialize lists to store true positive rate (TPR) and false positive rate (FPR)
        tpr_list = []
        fpr_list = []

        # Simulate clusters and evaluate performance
        for threshold in range(0, 20):  # Vary the threshold from 1 to 20
            signal_detected = 0
            background_detected = 0
            false_positives = 0

            for i in range(num_samples):
                cluster = self.cluster_simulator.generate_MIP_cluster()
                if self.is_signal(cluster, threshold):
                    signal_detected += 1
                else:
                    background_detected += 1

                    # Check false positives
                    if sum(cluster) > threshold:
                        false_positives += 1

            # Calculate rates
            signal_efficiency = signal_detected / num_samples
            background_efficiency = background_detected / num_samples
            false_positive_rate = false_positives / num_samples

            # Append rates to lists
            tpr_list.append(signal_efficiency)
            fpr_list.append(false_positive_rate)

        return tpr_list, fpr_list

    def plot_roc_curve(self, tpr_list, fpr_list):
        # Plot the ROC curve
        plt.figure()
        plt.plot(fpr_list, tpr_list, color='darkorange', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()


# Example of using the SimpleAlgo class
if __name__ == "__main__":
    simulator = ClusterSimulator("config1.json")
    algo = SimpleAlgo(simulator)
    tpr_list, fpr_list = algo.evaluate_performance()
    print("True Positive Rate:", tpr_list)
    print("False Positive Rate:", fpr_list)
    algo.plot_roc_curve(tpr_list, fpr_list)
