# Gaussian related functions
from tqdm import tqdm

import numpy as np
import tensorflow_probability as tfp

# It will return the list of mean and standard deviation for K gaussian.
def find_gaussian_parameters(data, K, n_iterations):
    # Here data is a one dimensional list
    samples = data.shape[0]

    # Initial guess of the parameters
    mus = np.random.rand(K)
    sigmas = np.random.rand(K)
    class_probabilities = np.random.dirichlet(np.ones(K))

    # Train the model
    for _ in tqdm(range(n_iterations), desc="Training for each pixel"):
        # E step
        resp = np.zeros((samples, K))
        for i in range(samples):
            for j in range(K):
                resp = class_probabilities * tfp.distributions.Normal(loc=mus[j], scale=sigmas[j]).prob(data[i])

        resp /= np.linalg.norm(resp, axis=1, ord=1, keepdims=True)
        class_resp = np.sum(resp, axis=0)

        # M step
        for i in range(K):
            class_probabilities[i] = class_probabilities[i]/samples
            mus[i] = np.sum(resp[:, i]*data)
            sigmas[i] = np.sqrt(np.sum(resp[:, i] * (data - mus[i])**2) / class_resp[i])

    return class_probabilities, mus, sigmas
