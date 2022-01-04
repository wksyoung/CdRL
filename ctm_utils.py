import numpy as np
N = 5

def uptree(weights):
    s = weights / np.sum(weights)
    # sum_exp = np.sum(np.exp(weights))
    # s = np.exp(weights) / sum_exp
    seed = np.random.random()
    head = 0
    winner = None
    for i in range(len(s)):
        upper = np.sum(s[:(i+1)])
        if seed >= head and seed <= upper:
            winner = i
            break
        else:
            head = upper
    if winner is None:
        winner = np.argmax(weights)
    return winner

def sleeping_experts(x, u, weights, actor, identifier):
    intensity = np.sum(weights)
    if identifier.analysis(x, u) > 0.1:
        pmis = 0
        sv = np.zeros([N])
        for i in range(N):
            p = weights[i] / intensity
            if identifier.analysis(x, actor.action(x, i)) > 0.1:# 0.15
                pmis = pmis + p
                sv[i] = 1
            else:
                sv[i] = 0
        for i in range(N):
            ri = pmis / (1 + 0.03) - sv[i]
            weights[i] = weights[i] * (1+0.03)**ri
    return weights

