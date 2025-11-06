import math
import time
from random import random
import matplotlib.pyplot as plt


def generatePoissonArrivalTime(rate=1):
    return math.log(random()) / rate * -1


def model1(total_population=100):
    current_time = 0
    states = {"infected": 1}
    while states["infected"] < total_population:
        current_time += generatePoissonArrivalTime(
            1)  # intuitively,the mean time until absorption will be proportional to the contact rate, so in our simulation I just use a rate of 1, we should prove this in our submission

        # simulate event
        person_one_infected = random() < states["infected"] / total_population
        person_two_infected = random() < (states["infected"] - person_one_infected) / (total_population - 1)
        if person_one_infected + person_two_infected == 1:
            # intuitively the mean time until absorption is proportional to the infection rate, we can prove this with poisson theorem 7/8 and the statement above
            states["infected"] += 1

    return current_time


def model1Alternative(total_population=100):
    current_time = 0
    states = {"infected": 1}
    while states["infected"] < total_population:
        infected = states["infected"]
        healthy = total_population - infected
        # probability of picking different when there are x infected and y healthy is (2xy) / [(x + y) (x + y - 1)]
        probability = (2 * (infected * healthy)) / (total_population * (total_population - 1))
        # instead of interarrival times between contact events, we just generate them between infection events
        current_time += generatePoissonArrivalTime(probability)

        # simulate event
        states["infected"] += 1

    return current_time


def model2(total_population=10):
    current_time = 0
    recovery_intensity = 0.5
    # algo is highly sensitive to the ratio between total_population and recovery_intensity
    states = {"infected": 1}
    while states["infected"] > 0:
        infected = states["infected"]
        healthy = total_population - infected
        # probability of picking different when there are x infected and y healthy is (2xy) / [(tot) (x + y - 1)]
        p_infection = (2 * (infected * healthy)) / (total_population * (total_population - 1))
        total_recovery_intensity = infected * recovery_intensity
        # instead of interarrival times between contact events, we just generate them between infection events
        current_time += generatePoissonArrivalTime(p_infection + total_recovery_intensity)

        # simulate event
        if random() < p_infection / (p_infection + total_recovery_intensity):
            states["infected"] += 1
        else:
            states["infected"] -= 1

    return current_time


def model2_modified(total_population=10):
    current_time = 0
    starting_infected = 1
    recovery_intensity = 0.5
    states = {"infected": starting_infected, "infectable": total_population - starting_infected}
    while states["infected"] > 0:
        infected = states["infected"]
        infectable = states["infectable"]
        # probability of picking different when there are x infected and y healthy is (2xy) / [(tot) (x + y - 1)]
        p_infection = (2 * (infected * infectable)) / (total_population * (total_population - 1))
        total_recovery_intensity = infected * recovery_intensity
        # instead of interarrival times between contact events, we just generate them between infection events
        current_time += generatePoissonArrivalTime(p_infection + total_recovery_intensity)

        # simulate event
        if random() < p_infection / (p_infection + total_recovery_intensity):
            states["infected"] += 1
            states["infectable"] -= 1
        else:
            states["infected"] -= 1

    return current_time


def simulate(runs, algo):
    start_time = time.perf_counter()
    running_sum = 0
    outcomes = []
    for i in range(runs):
        result = algo()
        outcomes.append(result)
        running_sum += result

    print("Simulation took " + str(time.perf_counter() - start_time) + " seconds")
    print("Average: " + str(running_sum / runs))

    plt.hist(outcomes, bins=30, color='skyblue', edgecolor='black')
    plt.title(algo.__name__ + " " + str(runs) + " Trials")
    plt.xlabel('Outcome')
    plt.ylabel('Occurrences')
    plt.show()
    plt.close()
