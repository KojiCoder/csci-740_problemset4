import math
import time
from random import random
import matplotlib.pyplot as plt
from collections import defaultdict

import numpy as np
from scipy.stats import norm


def calcVar(sum_squares, sums, n):
    return (sum_squares - (pow(sums, 2) / n)) / (n - 1)


def calcCoVar(sum_product, sum_x, sum_y, n):
    return (sum_product - (sum_x * sum_y / n)) / (n - 1)


def generatePoissonArrivalTime(rate):
    return math.log(random()) / rate * -1


def generateStageCompletionTime(mean):  # uniform distribution for simplicity :D
    return random() * mean * 2


def DESAlgo(end_time=10000, cost=1):
    cycle_count = 0
    cycle_start_time = 0
    cycle_profit = 0
    cycle_profit_squares = 0
    cycle_time_squares = 0
    cycle_profit_time_product = 0
    total_profit = 0
    k = 2
    customer_rate = 2
    current_time = 0
    stage_means = [0.3, 0.1, 0.2, 0.3, 0.1]
    stage_probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
    states = {"stage": 0}  # stage 0 is waiting for clients, otherwise it represents stage n

    p_accept = (k - cost) / k  # probability of customer accepting

    while True:
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
            completion_time = generateStageCompletionTime(stage_means[states["stage"]])
            current_time += completion_time
            cycle_profit += cost * completion_time
            # handle end of cycle
            if states["stage"] == 0:
                cycle_count += 1
                cycle_time = current_time - cycle_start_time
                cycle_time_squares += cycle_time ** 2
                cycle_profit_squares += cycle_profit ** 2
                cycle_profit_time_product += cycle_profit * cycle_time
                total_profit += cycle_profit

                if current_time > end_time:
                    break
                cycle_profit = 0
                cycle_start_time = current_time

    sample_mean = total_profit / current_time

    sigma_z = (calcVar(cycle_profit_squares, total_profit, cycle_count)
               - (2 * calcCoVar(cycle_profit_time_product, total_profit, current_time, cycle_count))
               + (calcVar(cycle_time_squares, current_time, cycle_count) * pow(sample_mean, 2)))

    err_margin = norm.ppf(0.975) * math.sqrt(sigma_z / cycle_count) / (current_time / cycle_count)

    return total_profit / current_time, err_margin


def DESAlgoStopping(cost=1, precision=0.01):
    cycle_count = 0
    cycle_start_time = 0
    cycle_profit = 0
    cycle_profit_squares = 0
    cycle_time_squares = 0
    cycle_profit_time_product = 0
    total_profit = 0
    k = 2
    customer_rate = 2
    current_time = 0
    stage_means = [0.3, 0.1, 0.2, 0.3, 0.1]
    stage_probabilities = [0.2, 0.2, 0.2, 0.2, 0.2]
    states = {"stage": 0}  # stage 0 is waiting for clients, otherwise it represents stage n

    p_accept = (k - cost) / k  # probability of customer accepting

    while True:
        if states["stage"] == 0:  # wait till customer arrival
            # we only need to consider customers that accept, so we can use customer_rate * p_accept
            current_time += generatePoissonArrivalTime(customer_rate * p_accept)

            u = random()
            sum_prob = stage_probabilities[0]
            while u > sum_prob:
                states["stage"] += 1
                sum_prob += stage_probabilities[states["stage"]]
            states["stage"] += 1
            if states["stage"] > 5:
                print("uh oh")
        else:
            states["stage"] -= 1
            completion_time = generateStageCompletionTime(stage_means[states["stage"]])
            current_time += completion_time
            cycle_profit += cost * completion_time
            total_profit += cost * completion_time
            # handle end of cycle
            if states["stage"] == 0:
                cycle_count += 1
                cycle_time = current_time - cycle_start_time
                cycle_time_squares += cycle_time ** 2
                cycle_profit_squares += cycle_profit ** 2
                cycle_profit_time_product += cycle_profit * cycle_time
                if cycle_count > 10:
                    sample_mean = total_profit / current_time

                    sigma_z = (calcVar(cycle_profit_squares, total_profit, cycle_count)
                               - (2 * calcCoVar(cycle_profit_time_product, total_profit, current_time, cycle_count))
                               + (calcVar(cycle_time_squares, current_time, cycle_count) * pow(sample_mean, 2)))

                    err_margin = norm.ppf(0.975) * math.sqrt(max(sigma_z / cycle_count, 0)) / (current_time / cycle_count)
                    if err_margin < precision:
                        break
                cycle_profit = 0
                cycle_start_time = current_time

    return total_profit / current_time, err_margin


def SearchSpace(searches):
    x_positions = []
    stats = []

    costs = np.linspace(0.0, 2, searches)[1:-1]

    for c in costs:
        avg, err = DESAlgoStopping(cost=c)  # your function: returns (avg, err)

        stats.append({
            'med': float(avg),
            'q1': float(avg - err),
            'q3': float(avg + err),
            'whislo': float(avg - err),
            'whishi': float(avg + err),
            # whiskers optional; omit to hide them
            'label': f'{c:.2f}'
        })
        x_positions.append(float(c))

    fig, ax = plt.subplots()

    # widths scaled to x-range; patch_artist fills the boxes
    width = 0.03 * (np.ptp(costs) if searches > 1 else 1.0)
    ax.bxp(stats, positions=x_positions, widths=width,
           showfliers=False, vert=True, patch_artist=True)

    # subtle fill
    for patch in getattr(ax, 'artists', []):
        patch.set_alpha(0.6)

    ax.set_xlabel("Cost")
    ax.set_ylabel("Value")
    ax.set_title("Custom interval boxes at numeric x-positions")
    plt.show()


# stuff for testing code
def stageGen():
    stage = 0
    stage_probabilities = [0.1, 0.2, 0.2, 0.2, 0.3]
    u = random()
    sum_prob = stage_probabilities[0]
    while u > sum_prob:
        stage += 1
        sum_prob += stage_probabilities[stage]
    stage += 1
    return stage


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
