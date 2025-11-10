import math
import bisect
import time
import mpmath as mp
import numpy as np
from collections import defaultdict
from random import random
from scipy.stats import norm
from matplotlib import pyplot as plt


def calcVar(sum_squares, sums, n):
    return (sum_squares - (pow(sums, 2) / n)) / (n - 1)


def calcCoVar(sum_product, sum_x, sum_y, n):
    return (sum_product - (sum_x * sum_y / n)) / (n - 1)


def generatePoissonArrivalTime(rate):
    return math.log(random()) / rate * -1


def generateGammaServiceTime(shape, rate):
    total = 0.0
    for _ in range(shape):
        total += -math.log(random()) / rate
    return total


def arrivalEvent(states, event_times):
    if len(event_times) == 0:
        bisect.insort(event_times, (generateGammaServiceTime(3, 4), serviceEvent), key=lambda x: x[0])
    else:
        states["waiting_queue"] += 1
    bisect.insort(event_times, (generatePoissonArrivalTime(1), arrivalEvent), key=lambda x: x[0])


def serviceEvent(states, event_times):
    if states["waiting_queue"] > 0:
        states["waiting_queue"] -= 1
        # generate next service event
        bisect.insort(event_times, (generateGammaServiceTime(3, 4), serviceEvent), key=lambda x: x[0])


def DESAlgo(precision=0.1):
    cycle_count = 0
    start_time = 0
    current_time = 0
    sum_waiting = 0
    queue_length_sum = 0
    event_times = []
    y_t_sum = 0
    y_squared_sum = 0
    t_squared_sum = 0
    states = {"waiting_queue": 0}
    bisect.insort(event_times, (generatePoissonArrivalTime(1), arrivalEvent), key=lambda x: x[0])
    while True:
        event_t, func = event_times[0]
        del event_times[0]
        # update residual clocks
        for i in range(len(event_times)):
            future_t, future_event = event_times[i]
            event_times[i] = (future_t - event_t, future_event)
        sum_waiting += states["waiting_queue"] * event_t
        current_time += event_t

        # event simulation
        func(states, event_times)

        # handle end of cycle
        if (states["waiting_queue"] == 0) and len(event_times) == 1:
            cycle_count += 1
            cycle_length = current_time - start_time
            queue_length_sum += sum_waiting
            y_t_sum += sum_waiting * cycle_length
            y_squared_sum += sum_waiting * sum_waiting
            t_squared_sum += cycle_length * cycle_length

            # Stop simulation when the confidence interval is small enough
            if cycle_count > 100:
                sample_mean = queue_length_sum / current_time

                sigma_z = (calcVar(y_squared_sum, queue_length_sum, cycle_count)
                           - (2 * calcCoVar(y_t_sum, queue_length_sum, current_time, cycle_count))
                           + (calcVar(t_squared_sum, current_time, cycle_count) * pow(sample_mean, 2)))

                err_margin = norm.ppf(0.975) * math.sqrt(sigma_z / cycle_count) / (current_time / cycle_count)

                if err_margin < precision:
                    break
            sum_waiting = 0
            start_time = current_time
    return sample_mean


def POMAlgo(precision=0.1):
    cycle_sojourn = 0
    cycle_count = 0
    cycle_start_time = 0
    total_sojourn = 0
    prev_arrival_time = 0
    prev_departure_time = 0
    sojourn_squares = 0
    cycle_time_squares = 0
    customers = 0
    product_sum = 0
    while True:
        customers += 1
        arrival_time = prev_arrival_time + generatePoissonArrivalTime(1)
        departure_time = max(arrival_time, prev_departure_time) + generateGammaServiceTime(3, 4)
        sojourn = departure_time - arrival_time
        cycle_sojourn += sojourn
        total_sojourn += sojourn

        # handle end of cycle
        if arrival_time > prev_departure_time:
            cycle_count += 1
            cycle_length = departure_time - cycle_start_time
            sojourn_squares += pow(cycle_sojourn, 2)
            cycle_time_squares += pow(cycle_length, 2)
            product_sum += cycle_sojourn * cycle_length

            # Stop simulation when the confidence interval is small enough
            if cycle_count > 100:
                sample_mean = total_sojourn / customers

                sigma_z = (calcVar(sojourn_squares, total_sojourn, cycle_count)
                           - (2 * calcCoVar(product_sum, total_sojourn, departure_time, cycle_count))
                           + (calcVar(cycle_time_squares, total_sojourn, cycle_count) * pow(sample_mean, 2)))

                err_margin = norm.ppf(0.975) * math.sqrt(max(sigma_z / cycle_count, 0)) / (departure_time / cycle_count)

                if err_margin < precision:
                    break
            cycle_sojourn = 0
            cycle_start_time = arrival_time

        prev_arrival_time = arrival_time
        prev_departure_time = departure_time

    return total_sojourn / customers


def POMAlgo(precision=0.1):
    cycle_sojourn = 0
    cycle_count = 0
    cycle_customers = 0
    total_sojourn = 0
    prev_arrival_time = 0
    prev_departure_time = 0
    sojourn_squares = 0
    cycle_customers_squares = 0
    customers = 0
    product_sum = 0
    while True:
        cycle_customers += 1
        customers += 1
        arrival_time = prev_arrival_time + generatePoissonArrivalTime(1)
        departure_time = max(arrival_time, prev_departure_time) + generateGammaServiceTime(3, 4)
        sojourn = departure_time - arrival_time
        cycle_sojourn += sojourn
        total_sojourn += sojourn

        # handle end of cycle
        if arrival_time > prev_departure_time:
            cycle_count += 1
            sojourn_squares += pow(cycle_sojourn, 2)
            cycle_customers_squares += pow(cycle_customers, 2)
            product_sum += cycle_sojourn * cycle_customers

            # Stop simulation when the confidence interval is small enough
            if cycle_count > 100:
                sample_mean = total_sojourn / customers

                sigma_z = (calcVar(sojourn_squares, total_sojourn, cycle_count)
                           - (2 * calcCoVar(product_sum, total_sojourn, departure_time, cycle_count))
                           + (calcVar(cycle_customers_squares, total_sojourn, cycle_count) * pow(sample_mean, 2)))

                err_margin = norm.ppf(0.975) * math.sqrt(max(sigma_z / cycle_count, 0)) / (departure_time / cycle_count)

                if err_margin < precision:
                    break
            cycle_sojourn = 0
            cycle_customers = 0

        prev_arrival_time = arrival_time
        prev_departure_time = departure_time

    return total_sojourn / customers


def RunDESAlgo(runs=20, precision=0.1):
    center = 1.5
    bin_width = 0.0125
    max_error = 0
    sample_means = []
    sigma = precision / norm.ppf(0.975)
    for i in range(runs):
        result = DESAlgo(precision)
        err = abs(result - center)
        if abs(result - center) > max_error:
            max_error = err
        sample_means.append(result)
    bins = np.arange(center - ((max_error // bin_width + 1) * bin_width),
                     center + max_error + bin_width, bin_width)
    counts, bin_edges, _ = plt.hist(sample_means, bins=bins, density=True, alpha=0.6)
    x = np.linspace(center - precision, center + precision, 1000)
    pdf = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    plt.plot(x, pdf, color='orange', label="Estimated Sampling Mean Distribution", linewidth=2)
    plt.axvline(center, color='red', linestyle="dashed", label="Population Mean")
    plt.axvline(center - precision, color='green', linestyle="dashed", label="Prediction Bound")
    plt.axvline(center + precision, color='green', linestyle="dashed")
    plt.title("Distribution of the Sample Mean of 20 DES Estimations")
    plt.xlabel("Stationary Average Queue Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("p5_des_fig.png")
    plt.show()
    plt.close()


def RunPOMAlgo(runs=20, precision=0.1):
    center = 2.25
    bin_width = 0.0125
    max_error = 0
    sample_means = []
    sigma = precision / norm.ppf(0.975)
    for i in range(runs):
        result = POMAlgo(precision)
        err = abs(result - center)
        if abs(result - center) > max_error:
            max_error = err
        sample_means.append(result)
    bins = np.arange(center - ((max_error // bin_width + 1) * bin_width),
                     center + max_error + bin_width, bin_width)
    counts, bin_edges, _ = plt.hist(sample_means, bins=bins, density=True, alpha=0.6)
    x = np.linspace(center - precision, center + precision, 1000)
    pdf = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    plt.plot(x, pdf, color='orange', label="Estimated Sampling Mean Distribution", linewidth=2)
    plt.axvline(center, color='red', linestyle="dashed", label="Population Mean")
    plt.axvline(center - precision, color='green', linestyle="dashed", label="Prediction Bound")
    plt.axvline(center + precision, color='green', linestyle="dashed")
    plt.title("Distribution of the Sample Mean of 20 POM Estimations")
    plt.xlabel("Stationary Average Queue Length")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("p5_pom_fig.png")
    plt.show()
    plt.close()
