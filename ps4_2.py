import math
import time
from collections import defaultdict
from random import random
from scipy.stats import t
from matplotlib import pyplot as plt


def calcVar(sum_squares, sums, n):
    return ((sum_squares - (pow(sums, 2) / n)) / (n - 1)) / n  # with bessel's correction


def generateGraph(mean_list, margin_list):
    sample_size = []
    upper_bound = []
    lower_bound = []
    for i in range(len(mean_list)):
        upper_bound.append(mean_list[i] + margin_list[i])
        lower_bound.append(mean_list[i] - margin_list[i])
        sample_size.append(i+2)
    plt.plot(sample_size, mean_list, color="red", label="Sample Mean")
    plt.plot(sample_size, upper_bound, color="green", label="Upper Bound")
    plt.plot(sample_size, lower_bound, color="yellow", label="Lower Bound")
    plt.ylim(0, 25)
    plt.xlabel("Sample Size")
    plt.ylabel("Iterations")
    plt.title("Approaching 95% Confidence of <5% Error")
    plt.legend()
    plt.show()


def findZeros(initial_state=1000):
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


def simulate(initial_state=1000, precision=0.05):
    sample_mean_list = []
    start_time = time.perf_counter()
    running_sum = 0
    sum_squares = 0
    outcomes = defaultdict(int)
    runs = 1
    err_margin = 0
    sample_mean = 0
    sample_mean_list = []
    err_margin_list = []
    while True:
        outcome = findZeros(initial_state)
        outcomes[outcome] += 1
        running_sum += outcome
        sum_squares += outcome * outcome
        sample_mean = running_sum / runs
        if runs > 2:
            err_margin = t.ppf(0.975, runs - 1) * math.sqrt(calcVar(sum_squares, running_sum, runs))
            sample_mean_list.append(sample_mean)
            err_margin_list.append(err_margin)
            if err_margin < sample_mean * precision:
                break
        runs += 1
    print("Runs: " + str(runs))
    print("Avg: " + str(sample_mean))
    print(f"Estimation Interval: [{sample_mean - err_margin:.2f}, {sample_mean + err_margin:.2f}]")
    print("Simulation took " + str(time.perf_counter() - start_time) + " seconds")

    plt.bar(outcomes.keys(), outcomes.values(), 1, color='g')
    plt.title("Find Zeros Simulation")
    plt.xlabel('Outcome')
    plt.ylabel('Occurrences')
    plt.show()
    plt.close()
    generateGraph(sample_mean_list, err_margin_list)