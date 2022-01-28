# Gaussian related functions
from tqdm import tqdm

import numpy as np
import tensorflow_probability as tfp

# It will return the list of mean and standard deviation for K gaussian.
def find_gaussian_parameters(data, K, n_iterations):
    # Here data is a one dimensional list
    samples = data.shape[0]
    print(data.shape)

    # Initial guess of the parameters
    mus = np.random.rand(K)
    sigmas = np.random.rand(K)
    class_probabilities = np.random.dirichlet(np.ones(K))

    # Train the model
    for _ in tqdm(range(n_iterations), desc="Training for each pixel"):
        # E step
        #resp = np.zeros((samples, K))
        #for i in range(samples):
        #    for j in range(K):
        #        resp[i, j] = class_probabilities[j] * tfp.distributions.Normal(loc=mus[j], scale=sigmas[j]).prob(data[i])

        resp = tfp.distributions.Normal(loc=mus, scale=sigmas).prob(data.reshape(-1, 1)).numpy() * class_probabilities

        resp /= np.linalg.norm(resp, axis=1, ord=1, keepdims=True)
        class_resp = np.sum(resp, axis=0)

        # M step
        for i in range(K):
            class_probabilities[i] = class_resp[i]/samples
            mus[i] = np.sum(resp[:, i] * data) / class_resp[i]
            sigmas[i] = np.sqrt(
                np.sum(resp[:, i] * (data - mus[i])**2) / class_resp[i]
            )

    return class_probabilities, mus, sigmas

# Testing the function
class_probs_true = [0.6, 0.4]
mus_true = [2.5, 4.8]
sigmas_true = [0.6, 0.3]
random_seed = 42
n_samples = 1000
n_iterations = 100
n_classes = 2

# generate the data
univariate_gmm = tfp.distributions.MixtureSameFamily(
        mixture_distribution=tfp.distributions.Categorical(probs=class_probs_true),
        components_distribution=tfp.distributions.Normal(
            loc=mus_true,
            scale=sigmas_true,
        )
    )

dataset = univariate_gmm.sample(n_samples, seed=random_seed).numpy()
class_probs, mus, sigmas = find_gaussian_parameters(dataset, n_classes, n_iterations)

print(class_probs)
print(mus)
print(sigmas)