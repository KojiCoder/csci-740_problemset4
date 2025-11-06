import math
import bisect
import time
from collections import defaultdict
from random import random

from matplotlib import pyplot as plt
from collections import defaultdict


# import matplotlib.pyplot as plt
# from collections import defaultdict


def calcVar(sum_squares, sums, n):
    return ((sum_squares - (pow(sums, 2) / n)) / (n - 1)) / n  # with bessel's correction


def generatePoissonArrivalTime(rate):
    return math.log(random()) / rate * -1


def generateGammaServiceTime(shape, rate):
    total = 0.0
    for _ in range(shape):
        total += -math.log(random()) / rate
    return total


def customerArrival(states, event_times):
    states["queue_length"] += 1
    states["total_customers"] += 1
    bisect.insort(event_times, (generatePoissonArrivalTime(1), customerArrival), key=lambda x: x[0])
    if states["queue_length"] == 1:
        # generate the service time
        bisect.insort(event_times, (generateGammaServiceTime(3, 4), customerService), key=lambda x: x[0])


def customerService(states, event_times):
    states["queue_length"] -= 1
    if states["queue_length"] > 0:
        bisect.insort(event_times, (generateGammaServiceTime(3, 4), customerService), key=lambda x: x[0])


def DESAlgo(end_time):
    current_time = 0
    event_times = []
    bisect.insort(event_times, (generatePoissonArrivalTime(1), customerArrival), key=lambda x: x[0])
    states = {"queue_length": 0, "wait_time": 0, "total_customers": 0}
    while current_time < end_time:
        time, arrival = event_times[0]
        del event_times[0]
        for i in range(len(event_times)):
            t, g = event_times[i]
            event_times[i] = (t - time, g)
        if states["queue_length"] > 0:
            states["wait_time"] += (states["queue_length"] - 1) * time
        arrival(states, event_times)  # we simulate the event here
        current_time += time

    return states["wait_time"] / end_time


def POMAlgo(precision=0.1, confidence=0.05):
    total_sojourn = 0
    prev_arrival_time = 0
    prev_departure_time = 0
    sum_squares = 0
    customers = 0
    while True:
        customers += 1
        arrival_time = prev_arrival_time + generatePoissonArrivalTime(1)
        departure_time = max(arrival_time, prev_departure_time) + generateGammaServiceTime(3, 4)
        sojourn = departure_time - arrival_time
        total_sojourn += sojourn
        sum_squares += pow(sojourn, 2)

        prev_arrival_time = arrival_time
        prev_departure_time = departure_time

        # this method doesnt work!, we need to switch to regenerative method in the slides
        # area outside the curve is less than confidence / 2, for half the distribution
        # 1 - area below > confidence
        if customers > 100 and 1.96 * math.sqrt(calcVar(sum_squares, total_sojourn, customers)) < precision:
            break

    return total_sojourn / customers, customers


def simulate(runs, algo):
    start_time = time.perf_counter()
    outcomes = []
    for i in range(runs):
        outcomes.append(algo())

    print("Simulation took " + str(time.perf_counter() - start_time) + " seconds")

    plt.hist(outcomes, bins=30, color='skyblue', edgecolor='black')
    plt.title(algo.__name__ + " " + str(runs) + " Trials")
    plt.xlabel('Outcome')
    plt.ylabel('Occurrences')
    plt.show()
    plt.close()
