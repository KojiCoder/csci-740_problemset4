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

# calculates 95 percent confidence interval using t distribution
def calc5PercentErrorMargin(outcome_list, sample_mean, sample_size):
    sample_sd = math.sqrt(calcSampleVariance(outcome_list, sample_mean, sample_size))
    return (t.ppf(0.975, sample_size-1) * (sample_sd / math.sqrt(sample_size)))

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

def generateHist(outcome_list, sample_mean):
    plt.hist(outcome_list)
    mean_label = "Mean: " + str(f"{(sample_mean):.2f}")
    plt.axvline(sample_mean, color = 'r', label=mean_label)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Occurrences")
    plt.title("Distribution On The Number of Iterations")
    plt.legend()
    plt.show()
    
def generateGraph(mean_list, margin_list):
    sample_size = []
    upper_bound = []
    lower_bound = []
    for i in range(len(mean_list)):
        upper_bound.append(mean_list[i] + margin_list[i])
        lower_bound.append(mean_list[i] - margin_list[i])
        sample_size.append(i+2)
    plt.plot(sample_size, mean_list, color="red")
    plt.plot(sample_size, upper_bound, color="green", label="Upper Interval")
    plt.plot(sample_size, lower_bound, color="yellow", label="Lower Interval")
    plt.ylim(0, 25)
    plt.xlabel("Number of Sample Size")
    plt.ylabel("Sample Mean")
    plt.title("Approaching 95% Confidence Interval")
    plt.legend()
    plt.show()

def simulate():
    outcome_list = []
    sample_mean_list = []
    err_margin_list = []
    stop = False
    start_time = time.perf_counter()
    # needs to run the algorithm two times outside the loop and then append to the list
    for _ in range(2):
        outcome_list.append(findZeros(1000))
    while not stop:
        # run the algorithm once and append the outcome to the list
        outcome_list.append(findZeros(1000))
        # obtains the sample size so far
        sample_size = len(outcome_list)
        # calculates the sample mean so far
        sample_mean = calcSampleMean(outcome_list, sample_size)
        sample_mean_list.append(sample_mean)
        err_margin = calc5PercentErrorMargin(outcome_list, sample_mean, sample_size)
        err_margin_list.append(err_margin)
        if err_margin <= sample_mean * 0.05:
            stop = True
    end_time = time.perf_counter()
    # display info
    print("Runs: " + str(sample_size))
    print("Avg: " + str(f"{(sample_mean):.2f}"))
    lower_value = sample_mean - err_margin_list[len(err_margin_list) - 1]
    upper_value = sample_mean + err_margin_list[len(err_margin_list) - 1]
    print("Interval: [" + str(f"{(lower_value):.2f}") + ", " + str(f"{(upper_value):.2f}") + "]")
    print("Simulation took " + str(end_time - start_time) + " seconds")
    # generate plot
    generateHist(outcome_list, sample_mean)
    generateGraph(sample_mean_list, err_margin_list)
    