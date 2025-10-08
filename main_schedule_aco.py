from differentiable_simulator import Simulator, Agent
import networkx as nx
import matplotlib.pyplot as plt
import torch
from jobshop_utils import createEdgesJobShop


number_of_machines_per_task = 3
number_of_tasks_per_job = 3
number_of_jobs = 8
num_agents_batches = 5

edges = createEdgesJobShop(number_of_machines_per_task, number_of_tasks_per_job, number_of_jobs)


G = nx.DiGraph()
G.add_edges_from(edges)

pos = nx.shell_layout(G)
nx.draw_networkx(G, pos, with_labels=True, node_color='lightblue', node_size=5000, edge_color='gray', arrowsize=20, arrows=True)

for edge in G.edges:
    G.edges[edge]['pheromone'] = torch.tensor(0.5)
    G.edges[edge]['travel_cost'] = torch.rand(size=G.edges[edge]['pheromone'].shape).abs()
    
S = Simulator()
S.addGraph(G)
S.addAdjacencyMatrix()
S.addPheromoneMatrix()
S.addHeuristicMatrix()


agents = [Agent() for _ in range(number_of_jobs)]
for job in range(number_of_jobs):
    agents[job].node = 'initiate_job_'+str(job)
    agents[job].final_node = 'finish_job_'+str(job)
    S.addAgent(agents[job])

S.coordination_step = True
S.step()
# S.addAgent(agent)
# S.addAgent(agent2)
# agent.step(S)
# S.updatePheromoneMatrix(agent.visited_nodes)