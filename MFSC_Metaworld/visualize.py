from experiments.parse import get_args
from experiments.experiment import PpoKeypointExperiment

if __name__ == '__main__':
    args = get_args()
    experiment = PpoKeypointExperiment(args)
    experiment.visualize()