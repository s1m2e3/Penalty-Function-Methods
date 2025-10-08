def addInitialJob(number_of_jobs,edges):
    final_edges = []
    for jobs in range(number_of_jobs):
        for machine_task in edges:
            if 0 == machine_task[1]:
                    source = ('initiate_job_'+str(jobs))
                    target = ('machine_'+str(machine_task[0]),'task_'+str(machine_task[1]),'job_'+str(jobs))
                    final_edges.append((source,target))
    return final_edges

def addFinalJob(number_of_jobs,number_of_tasks_per_job,edges):
    final_edges = []
    for jobs in range(number_of_jobs):
        for machine_task2 in edges:                
            if number_of_tasks_per_job-1 == machine_task2[1]:
                source = ('machine_'+str(machine_task2[0]),'task_'+str(machine_task2[1]),'job_'+str(jobs))
                target = ('finish_job_'+str(jobs))
                final_edges.append((source,target))
    return final_edges

def createEdgesJobShop(number_of_machines_per_task, number_of_tasks_per_job, number_of_jobs):
    edges = []
    for i in range(number_of_machines_per_task):
        for j in range(number_of_tasks_per_job):
            edges.append((i,j))
    final_edges = []
    final_edges = final_edges+addInitialJob(number_of_jobs,edges)
    final_edges = final_edges+addFinalJob(number_of_jobs,number_of_tasks_per_job,edges)
    
    for jobs in range(number_of_jobs):
        for machine_task in edges:
            for machine_task2 in edges:
                if machine_task != machine_task2 and machine_task[1]+1 == machine_task2[1]:
                    source = ('machine_'+str(machine_task[0]),'task_'+str(machine_task[1]),'job_'+str(jobs))
                    target = ('machine_'+str(machine_task2[0]),'task_'+str(machine_task2[1]),'job_'+str(jobs))
                    final_edges.append((source,target))
                    
    return final_edges