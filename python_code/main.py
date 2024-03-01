"""
EECE5698 Project 2
Optimal-policy-of-Maze-path
Author: Shuhao Liu
This code was inspired by the code written and maintained by Tim Miller,
Professor of Artifical Intelligence at The University of Queensland, Brisbane/Meaanjin, Australia.
https://gibberblot.github.io/rl-notes/intro.html#code


"""
from gridworld import GridWorld
from policy_iteration import PolicyIteration
from tabular_policy import TabularPolicy
from tabular_value_function import TabularValueFunction
from value_iteration import ValueIteration
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle



if __name__ == "__main__":
    # This is a random matrix for example purposes.
    # Matrix is defined as 20x20 instead of 18x18 stated in the project description
    State_Matrix = \
        np.array([
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 1, 1, 1, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 5, 1, 1, 5, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 7, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 2, 1, 1, 0, 1, 1, 5, 1, 1, 0, 0, 0, 0, 7, 7, 0],
             [0, 7, 7, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0],
             [0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 7, 7, 0, 1, 1, 1, 0, 1, 0],
             [0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 5, 0],
             [0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 7, 1, 0],
             [0, 1, 7, 0, 1, 1, 0, 1, 1, 0, 7, 7, 0, 0, 0, 0, 1, 7, 1, 0],
             [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 7, 1, 0],
             [0, 7, 1, 0, 1, 1, 5, 1, 1, 7, 1, 1, 1, 1, 1, 1, 1, 7, 1, 0],
             [0, 1, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
             [0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 9, 1, 1, 1, 1, 1, 0],
             [0, 7, 7, 7, 1, 0, 1, 1, 5, 1, 1, 1, 1, 1, 1, 1, 5, 1, 1, 0],
             [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 7, 7, 1, 1, 1, 1, 1, 1, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    rewardlist = {5: -5, 7: -10, 9: 200}

    # Policy_iteration
    # Base Scenario
    # gridworld1 = GridWorld.create(State_Matrix, rewardlist)
    # gridworld1.noise = 0.02/3
    # gridworld1.discount_factor = 0.95
    # policy = TabularPolicy(default_action=gridworld1.LEFT)
    # iterations, values = PolicyIteration(gridworld1, policy).policy_iteration(max_iterations=100, theta=0.01)
    # print("Number of iterations until convergence: %d" % (iterations))
    # # gridworld1.visualise_policy(policy, title='Policy for Base Scenario')
    # gridworld1.visualise_value_function(values, title="Optimal Value Function for Base Scenario", grid_size=1.5)
    # # optimal_path = gridworld1.find_optimal_path(policy)
    # # gridworld1.plot_optimal_path(optimal_path, "Optimal Path for Base Scenario")

    #
    # # Large Stochasticity Scenario
    # gridworld2 = GridWorld.create(State_Matrix, rewardlist)
    # gridworld2.noise = 0.5/3
    # gridworld2.discount_factor = 0.95
    # policy = TabularPolicy(default_action=gridworld2.LEFT)
    # iterations, values = PolicyIteration(gridworld2, policy).policy_iteration(max_iterations=100, theta=0.01)
    # print("Number of iterations until convergence: %d" % (iterations))
    # gridworld2.visualise_policy(policy, title='Optimal Policy for Large Stochasticity Scenario')
    # gridworld2.visualise_value_function(values, title="Optimal Value Function for Large Stochasticity Scenario", grid_size=1.5)
    # optimal_path = gridworld2.find_optimal_path(policy)
    # gridworld2.plot_optimal_path(optimal_path, "Large Stochasticity Scenario")
    #
    # # Small Discount Factor Scenario
    # gridworld3 = GridWorld.create(State_Matrix, rewardlist)
    # gridworld3.noise = 0.02/3
    # gridworld3.discount_factor = 0.55
    # policy = TabularPolicy(default_action=gridworld3.LEFT)
    # iterations, values = PolicyIteration(gridworld3, policy).policy_iteration(max_iterations=100, theta=0.01)
    # print("Number of iterations until convergence: %d" % (iterations))
    # gridworld3.visualise_policy(policy, title='Optimal Policy for Small Discount Factor Scenario')
    # gridworld3.visualise_value_function(values, title="Optimal Value Function for Small Discount Factor Scenario", grid_size=1.5)
    # optimal_path = gridworld3.find_optimal_path(policy)
    # gridworld3.plot_optimal_path(optimal_path, "Small Discount Factor Scenario")

    # Value iteration
    # Base Scenario
    gridworldv1 = GridWorld.create(State_Matrix, rewardlist)
    gridworldv1.noise = 0.02/3
    gridworldv1.discount_factor = 0.95
    values = TabularValueFunction()
    ValueIteration(gridworldv1, values).value_iteration(max_iterations=4)
    gridworldv1.visualise_value_function(values, "Value function for Base Scenario(Value iteration)")
    policy = values.extract_policy(gridworldv1)
    gridworldv1.visualise_policy(policy, "Policy for Base Scenario(Value iteration)")

    # # Large Stochasticity Scenario
    # gridworldv2 = GridWorld.create(State_Matrix, rewardlist)
    # gridworldv2.noise = 0.02/3
    # gridworldv2.discount_factor = 0.95
    # values = TabularValueFunction()
    # ValueIteration(gridworldv2, values).value_iteration(max_iterations=100)
    # gridworldv2.visualise_value_function(values, "Value function for Large Stochasticity Scenario(Value iteration)")
    # policy = values.extract_policy(gridworldv2)
    # gridworldv2.visualise_policy(policy, "Policy for Large Stochasticity Scenario(Value iteration)")
    #
    # # Small Discount Factor Scenario
    # gridworldv3 = GridWorld.create(State_Matrix, rewardlist)
    # gridworldv3.noise = 0.02/3
    # gridworldv3.discount_factor = 0.95
    # values = TabularValueFunction()
    # ValueIteration(gridworldv3, values).value_iteration(max_iterations=100)
    # gridworldv3.visualise_value_function(values, "Value function for Small Discount Factor Scenario(Value iteration)")
    # policy = values.extract_policy(gridworldv3)
    # gridworldv3.visualise_policy(policy, "Policy for Small Discount Factor Scenario(Value iteration)")
