from differentiable_simulator import Simulator, Agent
import networkx as nx
import matplotlib.pyplot as plt
import torch
from jobshop_utils import createEdgesJobShop


number_of_machines_per_task = 4
number_of_tasks_per_job = 3
number_of_jobs = 10
num_agents_batches = 5

edges = createEdgesJobShop(number_of_machines_per_task, number_of_tasks_per_job, number_of_jobs)

G = nx.DiGraph()
G.add_edges_from(edges)

for edge in G.edges:
    G.edges[edge]['pheromone'] = torch.tensor(0.5)
    G.edges[edge]['travel_cost'] = torch.rand(size=G.edges[edge]['pheromone'].shape).abs()*10
    
S = Simulator()
S.addGraph(G)
S.addAdjacencyMatrix()
S.addPheromoneMatrix()
S.addHeuristicMatrix()
S.num_jobs = number_of_jobs
S.num_tasks = number_of_tasks_per_job
S.num_machines = number_of_machines_per_task
S.target_node = 'finish_task_'+str(number_of_tasks_per_job-1)

agents = [Agent() for _ in range(num_agents_batches)]
for agent in agents :
    agent.node = 'initiate_task_'+str(0)
    agent.final_node = 'finish_task_'+str(number_of_tasks_per_job-1)
    agent.accumulatedTimeVector(number_of_jobs,number_of_tasks_per_job)
    S.addAgent(agent)

S.coordination_step = False
S.runOneInnerLoop()
# S.addAgent(agent)
# S.addAgent(agent2)
# agent.step(S)
# S.updatePheromoneMatrix(agent.visited_nodes)