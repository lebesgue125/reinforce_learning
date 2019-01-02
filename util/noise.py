import numpy as np


def ou_generate_noise(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)
