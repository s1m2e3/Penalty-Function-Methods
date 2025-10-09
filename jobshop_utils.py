def addInitialJob(num_machines):
    final_edges = []
    for machines in range(num_machines):
        source = ('initiate_task_'+str(0))
        target = ('task_'+str(0),'machine_'+str(machines),'job_'+str(0))
        final_edges.append((source,target))
    return final_edges

def addFinalJob(num_of_jobs,num_tasks_per_job,num_machines):
    final_edges = []
    for machines in range(num_machines):
        target = ('finish_task_'+str(num_tasks_per_job-1))
        source = ('task_'+str(num_tasks_per_job-1),'machine_'+str(machines),'job_'+str(num_of_jobs-1))
        final_edges.append((source,target))
    return final_edges
    
def createEdgesJobShop(number_of_machines_per_task, number_of_tasks_per_job, number_of_jobs):
    edges = []
    for i in range(number_of_tasks_per_job):
        for j in range(number_of_machines_per_task):
            for k in range(number_of_jobs):
                edges.append((i,j,k))
            
    final_edges = []
    final_edges = final_edges+addInitialJob(number_of_machines_per_task)
    final_edges = final_edges+addFinalJob(number_of_jobs,number_of_tasks_per_job,number_of_machines_per_task)
    
    for node in edges:
        for node2 in edges:
            if node != node2:
                if node[0] == node2[0] and node[2]+1 == node2[2]:
                    source = ('task_'+str(node[0]),'machine_'+str(node[1]),'job_'+str(node[2]))
                    target = ('task_'+str(node2[0]),'machine_'+str(node2[1]),'job_'+str(node2[2]))
                    final_edges.append((source,target))
                if node[0]+1==node2[0] and node[2] == number_of_jobs-1 and node2[2]==0:
                    source = ('task_'+str(node[0]),'machine_'+str(node[1]),'job_'+str(node[2]))
                    target = ('task_'+str(node2[0]),'machine_'+str(node2[1]),'job_'+str(node2[2]))
                    final_edges.append((source,target))
                    
    return final_edges