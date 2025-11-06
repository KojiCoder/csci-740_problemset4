import math
import time
from random import random
import matplotlib.pyplot as plt
from collections import defaultdict


def generatePoissonArrivalTime(rate):
    return math.log(random()) / rate * -1


def generateStageCompletionTime(mean):  # uniform distribution for simplicity :D
    return random() * mean * 2


def DESAlgo(end_time=10000, cost=1, price=1):
    total_working_time = 0
    k = 2
    customer_rate = 2
    current_time = 0
    stage_means = [0.3, 0.1, 0.2, 0.3, 0.1]
    stage_probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
    states = {"stage": 0}  # stage 0 is waiting for clients, otherwise it represents stage n

    p_accept = (k - cost) / k  # probability of customer accepting

    while current_time < end_time:

        if states["stage"] == 0:  # wait till customer arrival
            # we only need to consider customers that accept, so we can use customer_rate * p_accept
            current_time += generatePoissonArrivalTime(customer_rate * p_accept)

            u = random()
            sum_prob = stage_probabilities[0]
            while u > sum_prob:
                states["stage"] += 1
                sum_prob += stage_probabilities[states["stage"]]
            states["stage"] += 1
        else:
            states["stage"] -= 1
            completionTime = generateStageCompletionTime(stage_means[states["stage"]])
            current_time += completionTime
            total_working_time += completionTime

    return cost * total_working_time


# we search the whole space so we get a nice graph, also we need to compare the confidence intervals not just the means

# we can also do a binary search to get a more precise value, running more simulations until the two confidence intervals do not overlap anymore


# stuff for testing code
def stageGen():
    states = {"stage": 0}
    stage_probabilities = [0.1, 0.2, 0.2, 0.2, 0.3]
    u = random()
    sum_prob = stage_probabilities[0]
    while u > sum_prob:
        states["stage"] += 1
        sum_prob += stage_probabilities[states["stage"]]
    states["stage"] += 1
    return states["stage"]


def simulateStage(runs):
    start_time = time.perf_counter()
    running_sum = 0
    outcomes = defaultdict(int)
    for i in range(runs):
        outcomes[stageGen()] += 1

    for key, value in outcomes.items():
        running_sum += key * value
        # print(f"{key}: {value}")

    print("Avg: " + str(running_sum / runs))
    print("Simulation took " + str(time.perf_counter() - start_time) + " seconds")

    plt.bar(outcomes.keys(), outcomes.values(), 1, color='g')
    plt.title("StageGen " + str(runs) + " Trials")
    plt.xlabel('Outcome')
    plt.ylabel('Occurrences')
    plt.show()
    plt.close()
