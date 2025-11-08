import math
import time
from collections import defaultdict
from random import random
from scipy.stats import t
from matplotlib import pyplot as plt

def calcVar(sum_squares, sums, n):
    return ((sum_squares - (pow(sums, 2) / n)) / (n - 1)) / n  # with bessel's correction

# calculates the sampling mean of the sampling distribution
def calcSampleMean(outcome_list, sample_size):
    summation_result = 0
    for outcome in outcome_list:
        summation_result += outcome
    return float(summation_result)/sample_size

# calculates the sample variance
def calcSampleVariance(outcome_list, sample_mean, sample_size):
    summation_result = 0
    for outcome in outcome_list:
        summation_result += pow((outcome - sample_mean), 2)
    return float(summation_result)/(sample_size - 1)

# calculates the bound interval that would contain the true mean mu
def calc5PercentErrorMean(outcome_list, sample_mean, sample_size, bound):
    sample_sd = math.sqrt(calcSampleVariance(outcome_list, sample_mean, sample_size))
    _range = 1.96 * (sample_sd / math.sqrt(sample_size))
    if bound == "lower":
        return sample_mean - _range
    elif bound == "upper":
        return sample_mean + _range

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

# def findZeros(m):
#     iterations = 1
#     j = m
#     prob = random()
#     # checks if it can find zero next step
#     if prob < 1/j:
#         return iterations
#     prob -= 1/j
#     for k in range(j-1, 0, -1):
#         state_prob = 2*k/pow(j,2)
#         if prob > state_prob:
#             prob -= state_prob
#         else:
#             return iterations + findZeros(k)

# def simulate(initial_state=1000, precision=1):
#     start_time = time.perf_counter()
#     running_sum = 0
#     sum_squares = 0
#     outcomes = defaultdict(int)
#     runs = 1
#     while True:
#         outcome = findZeros(initial_state)
#         outcomes[outcome] += 1
#         running_sum += outcome
#         sum_squares += outcome * outcome
#         # todo: go from fixed interval to relative error
#         if runs > 2 and t.ppf(0.975, runs-1) * math.sqrt(calcVar(sum_squares, running_sum, runs)) < precision:
#             break
#         runs += 1
#     print("Runs: " + str(runs))
#     print("Avg: " + str(running_sum / runs))
#     print("Simulation took " + str(time.perf_counter() - start_time) + " seconds")

#     plt.bar(outcomes.keys(), outcomes.values(), 1, color='g')
#     plt.title("Find Zeros Simulation")
#     plt.xlabel('Outcome')
#     plt.ylabel('Occurrences')
#     plt.show()
#     plt.close()

def generateGraph(outcome_list, sample_mean):
    plt.hist(outcome_list)
    mean_label = "Mean: " + str(f"{(sample_mean):.2f}")
    plt.axvline(sample_mean, color = 'r', label=mean_label)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Occurrences")
    plt.legend()
    plt.show()

def simulate(precision=1):
    outcome_list = []
    stop = False
    start_time = time.perf_counter()
    # needs to run the algorithm once outside the loop and then append to the list
    outcome_list.append(findZeros(1000))
    while not stop:
        # run the algorithm once and append the outcome to the list
        outcome_list.append(findZeros(1000))
        # obtains the sample size so far
        sample_size = len(outcome_list)
        # calculates the sample mean so far
        sample_mean = calcSampleMean(outcome_list, sample_size)
        # calculates lower and upper bound
        lower_mu = calc5PercentErrorMean(outcome_list, sample_mean, sample_size, "lower")
        upper_mu = calc5PercentErrorMean(outcome_list, sample_mean, sample_size, "upper")
        est_interval = upper_mu - lower_mu
        if (sample_size > 2) and (est_interval < precision):
            stop = True
    end_time = time.perf_counter()
    # display info
    print("Runs: " + str(sample_size))
    print("Avg: " + str(sample_mean))
    print("Simulation took " + str(end_time - start_time) + " seconds")
    # generate plot
    generateGraph(outcome_list, sample_mean)