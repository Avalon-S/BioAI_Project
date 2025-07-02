import numpy as np
import matplotlib.pyplot as plt

def visualize_schedule(solution, processing_times, num_machines, makespan=None, load_balance=None, idle_time=None, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab20(np.linspace(0, 1, len(solution)))

    machine_available_time = {i: 0 for i in range(num_machines)}
    job_last_end_time = [0] * len(solution)

    for job_idx, job_schedule in enumerate(solution):
        for op_idx, machine_idx in enumerate(job_schedule):
            op_time = processing_times[job_idx, op_idx, machine_idx]
            if np.isfinite(op_time):
                start_time = max(machine_available_time[machine_idx], job_last_end_time[job_idx])
                end_time = start_time + op_time

                ax.barh(
                    y=machine_idx, width=op_time, left=start_time, height=0.8,
                    color=colors[job_idx],
                    label=f'Job {job_idx + 1}' if op_idx == 0 else ""
                )

                ax.text(
                    start_time + op_time / 2, machine_idx, f"{job_idx + 1}-{op_idx + 1}",
                    ha='center', va='center', rotation=90, fontsize=10, color='white'
                )

                machine_available_time[machine_idx] = end_time
                job_last_end_time[job_idx] = end_time

    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"Machine {i + 1}" for i in range(num_machines)])
    ax.set_xlabel("Time")
    ax.set_ylabel("Machines")
    ax.legend(loc="upper right", ncol=2, fontsize="small")

    title = "Job Scheduling Visualization"
    if makespan is not None and load_balance is not None and idle_time is not None:
        title += f" (Makespan: {makespan:.1f}, Load Balance: {load_balance:.1f}, Idle Time: {idle_time:.1f})"
    plt.title(title)

    if save_path:
        plt.savefig(save_path)
    plt.show()
    plt.close()