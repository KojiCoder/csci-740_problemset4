import math
import time
from collections import defaultdict
from random import random
from scipy.stats import t
from matplotlib import pyplot as plt


def calcVar(sum_squares, sums, n):
    return ((sum_squares - (pow(sums, 2) / n)) / (n - 1)) / n  # with bessel's correction


def findZeros(initial_state=100):
    iterations = 1
    prev_state = initial_state
    while True:
        iterations += 1

        u = random()
        cum_prob = 1 / prev_state

        if u < cum_prob:  # finding the zero with 1/j probability
            break

        # simulating k, the state of the current step
        new_state = prev_state - 1
        cum_prob += 2 * new_state / pow(prev_state, 2)

        while cum_prob < u:
            new_state -= 1
            cum_prob += 2 * new_state / pow(prev_state, 2)
        prev_state = max(new_state, 1)
    return iterations


def simulate(initial_state=1000, precision=1):
    start_time = time.perf_counter()
    running_sum = 0
    sum_squares = 0
    outcomes = defaultdict(int)
    runs = 1
    while True:
        outcome = findZeros(initial_state)
        outcomes[outcome] += 1
        running_sum += outcome
        sum_squares += outcome * outcome
        # todo: go from fixed interval to relative error
        if runs > 2 and t.ppf(0.975, runs-1) * math.sqrt(calcVar(sum_squares, running_sum, runs)) < precision:
            break
        runs += 1
    print("Runs: " + str(runs))
    print("Avg: " + str(running_sum / runs))
    print("Simulation took " + str(time.perf_counter() - start_time) + " seconds")

    plt.bar(outcomes.keys(), outcomes.values(), 1, color='g')
    plt.title("Find Zeros Simulation")
    plt.xlabel('Outcome')
    plt.ylabel('Occurrences')
    plt.show()
    plt.close()
