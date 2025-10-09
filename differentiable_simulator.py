import torch
import networkx as nx

def hard_deterministic_gumbel_softmax(logits, tau=1.0, dim=-1):
    y_soft = torch.nn.functional.softmax(logits / tau, dim=dim)
    index = y_soft.argmax(dim=dim, keepdim=True)
    y_hard = torch.zeros_like(logits).scatter_(dim, index, 1.0)
    return (y_hard - y_soft).detach() + y_soft


class Simulator():

    def __init__(self):
        self.agents = []
        self.graph = None
        self.evaporation_rate = 0.1
        self.target_node = None
        self.source_node = None
        self.Q = 1
        self.coordination_step = True
        self.processing_times = 5.0
        self.gradient_step = 0.5
        self.pheromone_weight = torch.nn.Parameter(torch.ones(1),requires_grad=True)
        self.heuristic_weight = torch.nn.Parameter(torch.ones(1),requires_grad=True)
    
    def runOneInnerLoop(self):
        
        agent_nodes = [self.target_node != agent.node for agent in self.agents]
        while any(agent_nodes):
            self.step()
            agent_nodes = [self.target_node != agent.node for agent in self.agents]
        arrivals = []
        for agent in self.agents:
            arrivals.append(self.applyConstraintArrivalTimes(agent.visited_nodes,agent.accumulated_time))
        arrivals = torch.stack(arrivals,dim=2)
        index = torch.argmin(arrivals.max(dim=1).values.max(dim=0).values).item()
        print(arrivals[:,:,index])
        # print(arrivals.shape)
        input('hipi')

    def estimateConstraintViolations(self,tuples,arrival_times_sorted):
        matrix = torch.zeros((len(tuples),1))
        for row_index in range(matrix.shape[0]):
            first_element_arrival_time = torch.where(arrival_times_sorted[:,2] == tuples[row_index][0])[0]
            second_element_arrival_time = torch.where(arrival_times_sorted[:,2] == tuples[row_index][1])[0]
            matrix[row_index] = matrix[row_index] - arrival_times_sorted[:,1][first_element_arrival_time]-self.processing_times+arrival_times_sorted[:,1][second_element_arrival_time]
        return matrix
    def applyGradientDescent(self,arrival_times_sorted,constraints_matrix):
        constraint_violation = torch.nn.functional.softplus(-constraints_matrix).sum()
        update = self.gradient_step*(torch.autograd.grad(constraint_violation,arrival_times_sorted,create_graph=True)[0])
        return update
    
    def optimize(self,arrival_times_sorted,tuples):
        while True:
            constraints_matrix = self.estimateConstraintViolations(tuples,arrival_times_sorted)
            if (-constraints_matrix>0).any().item():
                violated_constraints = torch.where(-constraints_matrix>0)
                arrival_times_sorted[1:,1]=arrival_times_sorted[1:,1]-self.applyGradientDescent(arrival_times_sorted,constraints_matrix[violated_constraints[0],violated_constraints[1]])[1:,1]
            else:
                break
        return arrival_times_sorted
    def updateArrivalTime(self,rows,arrival_times):
        
        sorted_times,indexes = torch.sort(arrival_times,dim=0)
        arrival_times_sorted = torch.stack((torch.tensor(rows),sorted_times,torch.arange(len(rows))),dim=1)
        tuples = []
        for row in arrival_times_sorted[:,2].tolist():
            for row2 in arrival_times_sorted[:,2].tolist():
                if row != row2 and (row,row2) not in tuples and (row2,row) not in tuples and row +1 == row2:
                    tuples.append((row,row2))
        
        updated_arrival_times_sorted = self.optimize(arrival_times_sorted,tuples)
        
        return updated_arrival_times_sorted[:,1]-arrival_times

    def collectDictionaryFromRoute(self,route):
        route_to_idx = [self.idx_to_node[i] for i in route if type(self.idx_to_node[i])!=str]
        tasks = set([node[0] for node in route_to_idx])
        machines = set([node[1] for node in route_to_idx])
        ordered_assignments_per_task = {task:{machine:[int(node[2].replace("job_","")) for node in route_to_idx if node[0]==task and node[1]==machine] for machine in machines } for task in tasks}
        new_keys = [int(task.replace('task_','')) for task in tasks]
        ordered_assignments_per_task =  {new_k:v for k,v in ordered_assignments_per_task.items() for new_k in new_keys if str(new_k) in k}
        
        return ordered_assignments_per_task
    def applyConstraintArrivalTimes(self,route,arrival_times):
        
        
        ordered_assignments_per_task = self.collectDictionaryFromRoute(route)
        for task in range(arrival_times.shape[1]):
        # scores = torch.softmax(torch.stack(scores),dim=1)
        # arrival_times = torch.stack(arrival_times)
        # one_hot_scores = hard_deterministic_gumbel_softmax(scores)
        # conflicting_machines = (one_hot_scores.sum(dim=0)>1).nonzero().squeeze()
        # cols = one_hot_scores[:, conflicting_machines]              # [num_jobs, 3]
        # indexes_conflicting = torch.stack(torch.where((cols == 1)),dim=1)
        # for i in set(indexes_conflicting[:,1].tolist()):
        #     relevant_jobs= torch.where(indexes_conflicting[:,1] == i)[0].tolist()
            for machine in ordered_assignments_per_task[task]:
                arrival_time_updates = self.updateArrivalTime(ordered_assignments_per_task[task][machine],arrival_times[ordered_assignments_per_task[task][machine],task])
                arrival_times[ordered_assignments_per_task[task][machine],task] = arrival_times[ordered_assignments_per_task[task][machine],task]+arrival_time_updates
                if task < arrival_times.shape[1]-1:
                    arrival_times[ordered_assignments_per_task[task][machine],task+1] = arrival_times[ordered_assignments_per_task[task][machine],task+1]+\
                                                                                        arrival_times[ordered_assignments_per_task[task][machine],task]+self.processing_times
        
        
        return arrival_times    

    def updateArrivalTimesAgents(self,agents,arrival_times):
        for i in range(len(agents)):
            agents[i].accumulated_time = arrival_times[i]
            
    def step(self):
        if self.coordination_step:
            scores = []
            arrival_times = []
            for agent in self.agents:
                if agent.node == self.target_node:
                    continue
                else:
                    scores.append(agent.estimateScores(self))
                    arrival_times.append(agent.accumulated_time)
            updated_arrival_times = self.applyConstraintArrivalTimes(scores,arrival_times)
            self.updateArrivalTimesAgents(self.agents,updated_arrival_times)
        else:
            for agent in self.agents:
                if agent.node == self.target_node:
                    continue
                else:
                    agent.stepChoice(self,agent.estimateScores(self))
                    
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
        agent.visited_nodes.append(self.node_to_idx[agent.node])
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
                with torch.no_grad():
                    vals_pheromones[crow[source]:crow[source+1]][idx[0].item()]=vals_pheromones[crow[source]:crow[source+1]][idx[0].item()]+update
                
class Agent():
    def __init__(self):
        self.node = None
        self.accumulated_length = torch.tensor(0.0)
        self.visited_nodes = []
        self.pheromone_deposit = torch.nn.Parameter(torch.ones(1),requires_grad=True)
        self.accumulated_cost = torch.tensor(0.0)
        self.final_node = None

    def accumulatedTimeVector(self,num_jobs,num_tasks):
        self.accumulated_time = torch.zeros(size=(num_jobs,num_tasks),requires_grad=True)+torch.abs(torch.rand(size=(num_jobs,num_tasks)))

    def estimateScores(self,simulator):
        idx = simulator.node_to_idx[self.node]
        crow = simulator.adjacency_matrix.crow_indices()
        cols = simulator.adjacency_matrix.col_indices()
        vals_pheromones = simulator.pheromone_matrix.values()
        vals_heuristic = simulator.heuristic_matrix.values()
        visited_mask = torch.isin(cols[crow[idx]:crow[idx+1]],torch.tensor(self.visited_nodes))
        pheromones_vector = vals_pheromones[crow[idx]:crow[idx+1]]
        heuristic_vector = vals_heuristic[crow[idx]:crow[idx+1]]
        probabilities = pheromones_vector**simulator.pheromone_weight*heuristic_vector**simulator.heuristic_weight/\
                            (torch.sum(pheromones_vector**simulator.pheromone_weight*heuristic_vector**simulator.heuristic_weight)+1e-5)
        probabilities = probabilities.masked_fill(visited_mask,0.0)
        return probabilities

    def stepChoice(self,simulator,probabilities):
        idx,crow,cols = self.getSparseMatrixData(simulator)
        vals_heuristic = simulator.heuristic_matrix.values()
        heuristic_vector = vals_heuristic[crow[idx]:crow[idx+1]]
        choice = torch.nn.functional.gumbel_softmax(probabilities,dim=-1,hard=True)
        next_node = cols[crow[idx]:crow[idx+1]][choice.argmax().item()].item()
        self.accumulated_cost = self.accumulated_cost + (heuristic_vector*choice).sum()
        self.updateNode(simulator.idx_to_node[next_node],next_node)    

    def getSparseMatrixData(self,simulator):
        idx = simulator.node_to_idx[self.node]
        crow = simulator.adjacency_matrix.crow_indices()
        cols = simulator.adjacency_matrix.col_indices()
        return idx,crow,cols
    def step(self,simulator):
        probabilities = self.estimateScores(simulator)
        self.stepChoice(simulator,probabilities)

    def updateNode(self,node,node_to_idx):
        self.node = node
        self.visited_nodes.append(node_to_idx)
        