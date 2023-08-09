import numpy as np
import math

def diversity_order_q(examples, q, similarity_funct, num_samples):
    '''
    Calculates the similarity-sensitive diversity of order q from a set of sentences.

    See Entropy and Diversity: The Axiomatic Approach (https://arxiv.org/abs/2012.02113) Ex. 6.1.7

    Estimate by sampling q elements, computing group similarity, and taking the mean mu_q

    The diversity of order q is (mu_q)^(1/(1-q))

    q is a "viewpoint parameter". Higher q values de-emphasize rarer "species" (in the biological setting)
    :param examples:
    :param q: Order for the diversity calculation
    :param similarity_funct: Symmetric function for calculating *pairwise* similarity
    :param num_samples: Number of times we sample q sentences
    :return:
    '''
    assert q >=2, "Order q must be at least 2"
    assert len(examples) > q, "Number of examples must be > the order for diversity calculation"

    # sample q examples from the set of examples with replacement
    group_similarities = []

    for i in range(num_samples):
        samples = np.random.choice(examples, size=q, replace=True)

        # Z_0,1 * ... Z_0,q-1
        x_0 = samples[0]
        remainder = samples[1:]

        pairwise_sims = [similarity_funct(x_0, y) for y in remainder]

        group_sim = np.product(pairwise_sims)

        group_similarities.append(group_sim)

    # expected group similarity
    mu_q = sum(group_similarities) / len(group_similarities)

    return math.pow(mu_q, 1 / (1-q))
