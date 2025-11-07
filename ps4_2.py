from random import random


def findZeros(initial_state):
    iterations = 1
    prev_state = initial_state
    while True:
        iterations += 1

        u = random()
        cum_prob = 1 / prev_state

        if u < cum_prob:  # finding the zero with 1/j probability
            break

        # simulating k, the state of the current step
        new_state = 1

        while cum_prob < u:
            cum_prob += 2 * new_state / pow(prev_state, 2)
            new_state += 1
        prev_state = new_state
    return iterations


def simulate():
    # track running variance?
    pass
