import numpy as np
from numpy import sin, cos
from numberpartitioning import *
import networkx as nx

import random

import gurobipy as gp
from gurobipy import GRB

from typing import Tuple
import cvxpy as cp

import scipy as sp
import scipy.optimize as opt

import time

from scipy.linalg import sqrtm


def problemToGraph(nums):
    """
    Transforms a number partitioning problem to a networkx graph
    INPUT:
    - ```nums''': problem
    OUTPUT:
    - ```graph''': problem graph
    """
    n = len(nums)
    graph = nx.Graph()
    dict_node_vals = {}
    
    for i in range(n):
        graph.add_node(i)
        dict_node_vals[i] = nums[i]
        nx.set_node_attributes(graph, dict_node_vals, "values")
        
    # weight of edge between two nodes is product of those nodes
    graph.add_weighted_edges_from([(u,v,dict_node_vals[u]*dict_node_vals[v]) for u,v in nx.complete_graph(n).edges()])
    
    return graph

def set_difference(arr, algorithm):
    """
    INPUT: 
    - ```arr''': array of partition problems
    - ```algorithm''': desired method to use from numberpartitioning package
        - greedy
        - karmarkar_karp
        - complete_karmarkar_karp
        - XQAOA
        - gw
    OUTPUT: 
    - ```results''': array of differences between partitions
    
    """
    
    if algorithm not in [Gurobi_Solver, greedy, complete_greedy, karmarkar_karp, complete_karmarkar_karp, XQAOA, QAOA, Goemans_Williamson]:
        raise ValueError
    
    results = []

    if algorithm is complete_karmarkar_karp or algorithm is complete_greedy:
        print(algorithm)
        for i in range(len(arr)):
            # iterates through different values of n/m
            avgs = []
            for j in range(len(arr[i])):
                result = next(algorithm(arr[i][j]))
                avgs.append(abs(result.sizes[0]-result.sizes[1]))
            results.append(avgs)
        return results
    
    elif algorithm is XQAOA:
        print(algorithm)
        for i in range(len(arr)):
            # iterates through different values of n/m
            avgs = []
            for j in range(len(arr[i])):
                # takes 20 measurements using X=Y mixer
                start_time = time.time()
                _, avg, _ = XY(arr[i][j])
                avgs.append(avg)
                print("i:",i,"j:",j,"avg:", avg)
                print("--- %s seconds ---" % (time.time() - start_time))
            results.append(avgs)
        return results

    elif algorithm is QAOA:
        print(algorithm)
        for i in range(len(arr)):
            # iterates through different values of n/m
            avgs = []
            for j in range(len(arr[i])):
                # takes 20 measurements
                start_time = time.time()
                _, avg, _ = QAOA(arr[i][j])
                avgs.append(avg)
                print("i:",i+2,"j:",j+1,"avg:", avg)
                print("--- %s seconds ---" % (time.time() - start_time))
            results.append(avgs)
        return results

    elif algorithm is Goemans_Williamson or algorithm is Gurobi_Solver:
        print(algorithm)
        for i in range(len(arr)):
            # iterates through different values of n/m
            avgs = []
            for j in range(len(arr[i])):
                start_time = time.time()
                result = algorithm(arr[i][j])
                avgs.append(result)
                print("i:",i+2,"j:",j+1,"avg:", result)
                print("--- %s seconds ---" % (time.time() - start_time))
            results.append(avgs)
        return results

    else:
        print(algorithm)
        # greedy, karmarkar_karp
        for i in range(len(arr)):
        # iterates through different values of n/m
            avgs = []
            for j in range(len(arr[i])):
                # iterates through data points
                result = algorithm(arr[i][j])
                avgs.append(abs(result.sizes[0]-result.sizes[1]))
            results.append(avgs)
        return results


# SOLVERS

 ###########################################################################################################################

def Gurobi_Solver(problem):
    """
    Solution obtained from GUROBI solver used to benchmark other solutions.
    INPUT:
    - ```problem''': array of partition problem
    OUTPUT:
    - ```objective_value''': cost value of problem
    - ```colors''': separates the problem graph into optimal disjoint sets
    - ```set_difference''': the optimal difference between the two sets
    """
    graph = problemToGraph(problem)
    graph_nodes = graph_nodes = sorted(graph.nodes())
    graph_edges = [tuple(sorted(graph_edge)) for graph_edge in graph.edges()]

    node_Vars = {}
    edge_Constrs = {}

    model = gp.Model()
    #verbose = False
    #if not verbose:
    model.Params.LogToConsole = 0

    #model.setParam('Threads', 3)
    #model.setParam('Symmetry', 0)
    #model.setParam('PreQLinearize', 2)

    for node in graph_nodes:
        node_key = f"{node}"
        node_Vars[node_key] = model.addVar(name=node_key, vtype=GRB.INTEGER, lb=-1, ub=1)
        edge_Constrs[node_key] = model.addConstr(lhs=node_Vars[node_key] * node_Vars[node_key], sense=GRB.EQUAL, rhs=1,
                                                 name=f"Constraint for {node}.")

    objective_function = gp.quicksum(0.5 * graph[i][j]["weight"]*(1 -  node_Vars[f'{i}'] * node_Vars[f'{j}']) for i, j in graph_edges)

    model.setObjective(objective_function, GRB.MAXIMIZE)
    model.optimize()

    objective = model.getObjective()
    objective_value: float = objective.getValue()

    colors = np.array([node_Vars[f"{i}"].getAttr('X') for i in graph_nodes], dtype='int')
    set_difference = 0
    for i in range(len(colors)):
        set_difference += colors[i] * graph.nodes[i]["values"]
    set_difference = abs(set_difference)
    #return objective_value, colors, 
    return set_difference

###########################################################################################################################
def QAOA(problem, samples = 20):
    
    graph = problemToGraph(problem)
    costs = []
    for i in range(samples):
        angles = [np.random.uniform(0, np.pi), np.random.uniform(0, 2*np.pi)]
        res = opt.minimize(QAOA_cost, 
                           angles, 
                           graph, 
                           options={"maxiter":10000}, method = 'COBYLA')
        #reduced = reduce_angles(res.x, graph)
        costs.append(QAOA_cost(res.x, graph))
    average = np.average(costs)
    std = np.std(costs)
    return costs, average, std


def QAOA_cost(angles, graph):

    n = len(graph.nodes)
    m = len(graph.edges)

    beta = angles[0]
    gamma = -4*angles[1]
    
    weight_dict = nx.get_edge_attributes(graph, "weight")

    mc_cost = 0

    for _, edge in enumerate(graph.edges()):
        u,v = edge
        w_uv = weight_dict[edge]
        
        edf = list(graph.neighbors(u))
        edf.remove(v)
        
        e_terms = 1
        d_terms = 1
        
        triangle_1_terms=1
        triangle_2_terms=1
        
        for w_id in edf:
            edge_uw = tuple(sorted((w_id, u)))       
            weight_uw = weight_dict[edge_uw]
            
            edge_wv = tuple(sorted((w_id, v)))        
            weight_wv = weight_dict[edge_wv]

            e_terms *= cos(gamma * weight_wv)
            d_terms *= cos(gamma * weight_uw)
            
            triangle_1_terms *= cos(gamma * weight_uw + gamma * weight_wv)
            triangle_2_terms *= cos(gamma * weight_uw - gamma * weight_wv)
        
        mc_cost += w_uv * (0.5 + 0.25*(sin(4*beta)*sin(gamma*w_uv)*(e_terms + d_terms) + \
            (sin(2*beta)**2) * (triangle_1_terms - triangle_2_terms)))
    
    return  np.sqrt(-4*mc_cost + sum([*nx.get_node_attributes(graph, "values").values()])**2)

###########################################################################################################################
def XY(problem, samples = 20):
    
    graph = problemToGraph(problem)
    n = len(graph.nodes)
    m = len(graph.edges)
    costs = []
    for i in range(samples):
        angles = np.random.uniform(0, 2*np.pi, n+m)
        res = opt.minimize(alpha_beta_dupe, 
                           angles, 
                           graph, 
                           options={"maxiter":1000}, method = 'COBYLA')
        reduced = reduce_angles(res.x, graph)
        costs.append(alpha_beta_dupe(reduced, graph))
    average = np.average(costs)
    std = np.std(costs)
    return costs, average, std

def alpha_beta_dupe(angles, graph):
    """
    Duplicates betas to alphas for X=Y mixer
    """
    # num angles should be nodes + edges
    num_nodes = len(graph.nodes)
    new_angles = np.concatenate((angles[:num_nodes], angles[:num_nodes], angles[num_nodes:]), axis=None)
    return XQAOA(new_angles, graph)

def XQAOA(angles, graph):
    """
    Calculates the total cost for the graph of a number partitioning problem using the analytical
    expression for depth p=1.
    INPUT:
    - ```angles''': angles alphas, betas, and gammas
    - ```graph''': networkx graph encoding a number partitioning problem
    OUPUT:
    - cost value of the graph
    """
    n = len(graph.nodes) 
    
    assert len(angles) == 2*n + len(graph.edges), "Number of angles passed is not equal to " \
                                                                           "2 * Num_Nodes + Num_Edges."
    # extract angles alpha, beta, and gamma
    alphas = angles[:n]
    betas = angles[n:2*n]
    gamma = angles[2*n:]
    gammas = [-4*g for g in gamma]
    
    dict_node_alphas = {}
    dict_node_betas = {}
    dict_edge_gammas = {}
    edges = list(graph.edges())
    edges_reversed = [(v,u) for (u,v) in edges]
    
    weights = nx.get_edge_attributes(graph, "weight")
    
    for i in range(n):
        dict_node_alphas[i] = alphas[i]
        dict_node_betas[i] = betas[i]
        
        nx.set_node_attributes(graph, dict_node_alphas, "alphas")
        nx.set_node_attributes(graph, dict_node_betas, "betas")
    for i in range(len(edges)):
        dict_edge_gammas[edges[i]] = gammas[i]
        dict_edge_gammas[edges_reversed[i]] = gammas[i]
        
        weights[edges_reversed[i]] = weights[edges[i]]
        
    #########################################################################
    def term1(edge):
        term = cos(2*alphas[u]) * cos(2*alphas[v]) * sin(weights[(u,v)]*dict_edge_gammas[(u,v)])
        
        e_terms = cos(2*betas[u])*sin(2*betas[v])
        
        for w in neighbours:
            e_terms *= cos(dict_edge_gammas[(w,v)]*weights[(w,v)])

        d_terms = sin(2*betas[u])*cos(2*betas[v])
        
        for w in neighbours:
            d_terms *= cos(dict_edge_gammas[(u,w)]*weights[(u,w)])
  
        return  term * (e_terms + d_terms)
        
    def term2(edge):
        term = 0.5*sin(2*alphas[u])*sin(2*alphas[v])
            
        F1_terms = 1
        F2_terms = 1
        for f in neighbours:
            F1_terms *= cos(dict_edge_gammas[(u,f)]*weights[(u,f)] + dict_edge_gammas[(v,f)]*weights[(v,f)])
            F2_terms *= cos(dict_edge_gammas[(u,f)]*weights[(u,f)] - dict_edge_gammas[(v,f)]*weights[(v,f)])

        return term * (F1_terms + F2_terms)  
    
    def term3(edge):
        term = 0.5*cos(2*alphas[u])*sin(2*betas[u])*cos(2*alphas[v])*sin(2*betas[v])
        
        F1_terms = 1
        F2_terms = 1
        for f in neighbours:
            #if f>v and f>u:
            F1_terms *= cos(dict_edge_gammas[(u,f)]*weights[(u,f)] + dict_edge_gammas[(v,f)]*weights[(v,f)])
            F2_terms *= cos(dict_edge_gammas[(u,f)]*weights[(u,f)] - dict_edge_gammas[(v,f)]*weights[(v,f)])
        
        return term * (F1_terms - F2_terms)
    #########################################################################
    cost = 0

    for edge in edges:
        u,v = edge
        e_list = list(graph.neighbors(v))
        e_list.remove(u)
        e_set = set(e_list)
        
        neighbours = list(e_set)

        cost += 0.5*weights[edge] + 0.5*weights[edge]*(term1(edge) - term2(edge) + term3(edge))
    return np.sqrt(-4*cost + sum([*nx.get_node_attributes(graph, "values").values()])**2)
    #return cost



def reduce_angles(angles, graph):
    """
    INPUT:
    - ```angles''': 2n+m array with alpha, beta, and gamma angles
    - ```graph''': networkx problem graph
    OUTPUT:
    - array of reduced angles
    """
    
    def approx_equal(num1, num2, epsilon = 1e-3):
        diff = abs(num1 - num2)
        return diff <= epsilon

    n = len(graph.nodes())
    betas = angles[:n]
    gammas = angles[n:]
    
    for i in range(len(betas)):
        while betas[i] > np.pi:
            betas[i] -= np.pi
        while betas[i] < 0.0:
            betas[i] += np.pi
        
        if approx_equal(betas[i], np.pi/4):
            betas[i] = np.pi/4
        elif approx_equal(betas[i], 3*np.pi/4):
            betas[i] = 3*np.pi/4
    
    edges = list(graph.edges)
    
    for i in range(len(gammas)):
        if approx_equal(gammas[i], 0.0, 1e-1):
            gammas[i] = 0.0
        elif approx_equal(gammas[i], np.pi, 1e-2):
            gammas[i] = np.pi
        elif approx_equal(gammas[i], 2*np.pi, 1e-2):
            gammas[i] = 0.0
        elif approx_equal(gammas[i], 3*np.pi, 1e-2):
            gammas[i] = np.pi
        else:
            new_angles = np.concatenate((betas,betas,gammas))
            total_cost = XQAOA(new_angles, graph)
            old_cost = total_cost
            gammas[i] = 0.0
            u,v = edges[i]
            b_options = [np.pi/4, 3*np.pi/4]
            opt_b = [betas[u], betas[v]]
            
            for beta_u in b_options:
                for beta_v in b_options:
                    betas[u] = beta_u
                    betas[v] = beta_v
                    
                    new_angles = np.concatenate((betas,betas,gammas))
                    total_cost = XQAOA(new_angles, graph)
                    new_cost = total_cost
                    if approx_equal(old_cost, new_cost) or new_cost < old_cost:
                        opt_b = [beta_u, beta_v]
                        
            betas[u] = opt_b[0]
            betas[v] = opt_b[1]
            gammas[i] = 0.0
    return np.concatenate((betas,gammas))

 ###########################################################################################################################

def Goemans_Williamson(array, row=0, col=0, iters=20):
    """
    Takes an array and uses Goemans_Williamson solver's color array to return cost
    """
    G = problemToGraph(array)
    n = G.number_of_nodes()

    simulation_costs = []
    #start = time.time()
    for i in range(0,iters):
        np.random.seed(row + 50*col + 19*25*i)
        u = np.random.randn(n)
        
        cost, colors = gw_cost(G, u)
#         cost, colors = Goemans_Williamson2(G)
        if cost != float("inf") and cost != float("-inf"):
            sim_cost = abs(array @ colors)
            simulation_costs.append(sim_cost)
    #print("--- %s seconds ---" % (time.time() - start))

    return np.average(simulation_costs)


def gw_cost(graph, vec):
    """
    The Goemans-Williamson algorithm for solving the maxcut problem.
    Ref:
        Goemans, M.X. and Williamson, D.P., 1995. Improved approximation
        algorithms for maximum cut and satisfiability problems using
        semidefinite programming. Journal of the ACM (JACM), 42(6), 1115-1145
    INPUT:
    - ```problem''': array of partition problem
    OUTPUT:
    - difference between two sets
    """
    n = len(graph.nodes())
    
    # creates n x n symmetric matrix optimisation variable
    X = cp.Variable((n,n), symmetric=True)

    # creates constraints on X (positive semidefinite & symmetric)
    constraints = [X>>0]
    constraints += [
        X[i,i] == 1 for i in range (n)
    ]

    # algorithm: 
    #objective = sum(0.5*graph[i][j]["weight"]*(1-X[i,j]) for (i,j) in graph.edges())
    objective = sum([graph[i][j]["weight"]*(X[i,j]) for (i,j) in graph.edges()])
    
    # SDP relaxation
    prob = cp.Problem(cp.Minimize(objective), constraints)
    cost = prob.solve(max_iters = 2500)
    if cost == float("inf") or cost == float('-inf'):
        return cost, []
    else:
        x = sqrtm(X.value)
        colours = np.sign(x @ vec)
        return cost, colours
