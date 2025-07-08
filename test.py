import numpy as np


def boot_CI(data, n_resamples=1000, confidence_level=0.95, random_state=None):
    rng = np.random.default_rng(random_state)
    try:
        boot_means = np.array([
            np.mean(rng.choice(data, size=len(data), replace=True))
            for _ in range(n_resamples)
        ])
        lower_bound = np.percentile(
            boot_means, (1 - confidence_level) / 2 * 100)
        upper_bound = np.percentile(
            boot_means, (1 + confidence_level) / 2 * 100)
        n = len(data)
        sqrt_adjustment = np.sqrt(n / (n - 1)) if n > 1 else 1
        return lower_bound * sqrt_adjustment, upper_bound * sqrt_adjustment
    except Exception as e:
        print(f"Error in boot_CI: {e}")
        raise


if __name__ == "__main__":
    signal = np.array([1.2, 1.5, 1.3, 1.7, 1.6, 1.8, 1.1, 1.4, 1.5])
    ci = boot_CI(signal)
    print(f"Bootstrapped CI: {ci}")
