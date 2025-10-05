import torch
import networkx as nx

class Simulator():

    def __init__(self):
        self.agents = []
        self.graph = None
        self.evaporation_rate = 0.1
        self.target_node = None
        self.source_node = None
        self.Q = 1
    def step(self):
        for agent in self.agents:
            if agent.node == self.target_node:
                path = agent.visited_nodes
            agent.step(self.graph)
        self.evaporate()
    
    def addGraph(self, graph):
        self.graph = graph
        self.addNodeToIdx()
        self.addAdjacencyMatrix()

    def addNodeToIdx(self):
        self.node_to_idx = {n:i for i,n in enumerate(list(self.graph.nodes()))}
        self.idx_to_node = {i:n for i,n in enumerate(list(self.graph.nodes()))}

    def fromSparseScipyToSparseTensor(self,sparse_scipy_matrix):
        crow = torch.from_numpy(sparse_scipy_matrix.indptr)
        cols = torch.from_numpy(sparse_scipy_matrix.indices)
        vals = torch.from_numpy(sparse_scipy_matrix.data)
        return torch.sparse_csr_tensor(crow,cols,vals,dtype=torch.float,requires_grad=True)
    def addPheromoneMatrix(self):
        pheromone_matrix = nx.to_scipy_sparse_array(
        self.graph,
        nodelist=self.graph.nodes(),
        weight="pheromone",    # reads edge['cost'] as the matrix entry
        dtype=float,
        format="csr"
        )
        self.pheromone_matrix = self.fromSparseScipyToSparseTensor(pheromone_matrix)
    def addHeuristicMatrix(self):
        heuristic_matrix=nx.to_scipy_sparse_array(
        self.graph,
        nodelist=self.graph.nodes(),
        weight="travel_cost",    # reads edge['cost'] as the matrix entry
        dtype=float,
        format="csr"
        )
        self.heuristic_matrix = self.fromSparseScipyToSparseTensor(heuristic_matrix)
    def addAdjacencyMatrix(self):
        adjacency_matrix = nx.adjacency_matrix(self.graph)
        self.adjacency_matrix = self.fromSparseScipyToSparseTensor(adjacency_matrix)
    def addAgent(self, agent):
        self.agents.append(agent)

    def updatePheromoneMatrix(self,path):
        update = self.Q/len(path)
        crow = self.adjacency_matrix.crow_indices()
        cols = self.adjacency_matrix.col_indices()
        vals_pheromones = self.pheromone_matrix.values()
        edges = [(path[i],path[i+1]) for i in range(len(path)-1)]
        for edge in edges:
            source = edge[0]
            target = edge[1]
            idx = (cols[crow[source]:crow[source+1]] == target).nonzero(as_tuple=True)[0]
            if idx.numel():
                print(vals_pheromones[crow[source]:crow[source+1]][idx[0]])
                vals_pheromones[crow[source]:crow[source+1]][idx[0].item()]=vals_pheromones[crow[source]:crow[source+1]][idx[0].item()]+update
                print(vals_pheromones[crow[source]:crow[source+1]][idx[0]])
                input('hipi')
class Agent():
    def __init__(self):
        self.node = None
        self.accumulated_length = torch.tensor(0.0)
        self.visited_nodes = []
        self.pheromone_weight = torch.nn.Parameter(torch.ones(1),requires_grad=True)
        self.heuristic_weight = torch.nn.Parameter(torch.ones(1),requires_grad=True)
        self.pheromone_deposit = torch.nn.Parameter(torch.ones(1),requires_grad=True)
        self.accumulated_cost = torch.tensor(0.0)
   
    def step(self,simulator):
        idx = simulator.node_to_idx[self.node]
        crow = simulator.adjacency_matrix.crow_indices()
        cols = simulator.adjacency_matrix.col_indices()
        vals_pheromones = simulator.pheromone_matrix.values()
        vals_heuristic = simulator.heuristic_matrix.values()
        visited_mask = torch.isin(cols[crow[idx]:crow[idx+1]],torch.tensor(self.visited_nodes))
        pheromones_vector = vals_pheromones[crow[idx]:crow[idx+1]]
        heuristic_vector = vals_heuristic[crow[idx]:crow[idx+1]]
        probabilities = pheromones_vector**self.pheromone_weight*heuristic_vector**self.heuristic_weight/\
                            (torch.sum(pheromones_vector**self.pheromone_weight*heuristic_vector**self.heuristic_weight)+1e-5)
        probabilities = probabilities.masked_fill(visited_mask,0.0)
        choice = torch.nn.functional.gumbel_softmax(probabilities,dim=-1,hard=True)
        next_node = cols[crow[idx]:crow[idx+1]][choice.argmax().item()].item()
        self.visited_nodes.append(next_node)
        self.accumulated_cost = self.accumulated_cost + (heuristic_vector*choice).sum()
        self.updateNode(simulator.idx_to_node[next_node])    

    def updateNode(self,node):
        self.node = node
        self.visited_nodes.append(node)

G = nx.grid_2d_graph(5,5)
for edge in G.edges:
    G.edges[edge]['pheromone'] = torch.tensor(0.5)
    G.edges[edge]['travel_cost'] = torch.rand(size=G.edges[edge]['pheromone'].shape).abs()

    
S = Simulator()
S.addGraph(G)
S.addAdjacencyMatrix()
S.addPheromoneMatrix()
S.addHeuristicMatrix()
agent = Agent()
agent.node = (0,0)
agent.visited_nodes.append(S.node_to_idx[agent.node])
agent2 = Agent()
agent2.node = (0,0)
agent2.visited_nodes.append(S.node_to_idx[agent2.node])
S.addAgent(agent)
S.addAgent(agent2)
agent.step(S)
S.updatePheromoneMatrix(agent.visited_nodes)