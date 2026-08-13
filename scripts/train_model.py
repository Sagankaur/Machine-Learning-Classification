from system_threat_forecaster import ExperimentConfig, run_experiment


if __name__ == "__main__":
    metrics = run_experiment(ExperimentConfig())
    print(f"Validation accuracy: {metrics['validation_accuracy']:.4f}")
