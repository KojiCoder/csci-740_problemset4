from random import random


def findZeros(initial_state):
    iterations = 1
    prev_state = initial_state
    while True:
        iterations += 1

        if random() < 1 / prev_state:  # finding the zero with 1/j probability
            break

        # simulating k, the state of the current step
        u = random()
        new_state = 1
        cum_prob = 0
        while cum_prob < u:
            cum_prob += 2 * new_state / pow(prev_state, 2)
            new_state += 1
        prev_state = new_state
    return iterations


def simulate():
    # track running variance?
    pass
