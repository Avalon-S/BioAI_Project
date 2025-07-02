import numpy as np

def read_and_parse_fjsp_file(file_path):
    """
    Read and parse the FJSP data file, converting it into operation_data and processing_times format
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Replace special characters and clean empty lines
    lines = [line.replace("→", " ").strip() for line in lines if line.strip()]

    # Read metadata
    meta_data = list(map(float, lines[0].split()))
    num_jobs, num_machines = int(meta_data[0]), int(meta_data[1])

    operation_data = []
    max_operations = 0

    for line in lines[1:]:
        data = list(map(float, line.split()))
        num_operations = int(data[0])
        max_operations = max(max_operations, num_operations)

        job_operations = []
        idx = 1
        for _ in range(num_operations):
            num_machines_for_op = int(data[idx])
            idx += 1
            machine_time_pairs = []
            for _ in range(num_machines_for_op):
                machine = int(data[idx])
                time = data[idx + 1]
                machine_time_pairs.append((machine, time))
                idx += 2
            job_operations.append(machine_time_pairs)
        operation_data.append(job_operations)

    processing_times = np.full((num_jobs, max_operations, num_machines), np.inf)
    for job_idx, job_operations in enumerate(operation_data):
        for op_idx, machine_time_pairs in enumerate(job_operations):
            for machine, time in machine_time_pairs:
                processing_times[job_idx, op_idx, machine - 1] = time

    return operation_data, processing_times, num_jobs, num_machines