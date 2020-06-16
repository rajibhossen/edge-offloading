import numpy as np

# [name, cpu cycle for 1 bit, min data, maximum data]
# applications = [
#       [id, cpu cycle, min data, max data, delay tolerance]
#     [0, 10435, 80000, 800000, 5],  # speech recognition
#     [1, 25346, 300, 1400, 0.5],  # natural language processing
#     [2, 45043, 300000, 30000000, 300],  # face recognition
#     [3, 34252, 300, 4000, 0.1],  # language translation
#     [4, 54633, 100000, 3000000, 50],  # 3d game processing
#     [5, 40305, 100000, 3000000, 40],  # virtual reality
#     [6, 34532, 100000, 3000000, 40],  # augmented reality
# ]
applications = [
    [0, 10435, 440000.0, 8],  # speech recognition
    # [1, 25346, 850.0, 1.5],  # natural language processing, newly added
    [2, 45043, 15150000, 500],  # face recognition
    [3, 34252, 2150.0, 1],  # language translation
    [4, 54633, 1550000.0, 100],  # 3d game processing
    [5, 40305, 1550000.0, 80],  # virtual reality
    [6, 34532, 1550000.0, 70],  # augmented reality
    # [7, 100, 15500000.0, 20], # fabricated task, newly added
]


def get_random_task():
    choice = np.random.choice(len(applications))
    application = applications[choice]
    data_size = application[2]
    cpu_cycle = data_size * application[1]
    task = {
        "data": data_size,
        "cpu_cycle": cpu_cycle,
        "dt": application[3]
    }
    return task


def make_task_from_applications(application):
    data_size = application[2]
    cpu_cycle = data_size * application[1]
    task = {
        "data": data_size,
        "cpu_cycle": cpu_cycle,
        "dt": application[3]
    }
    return task


# def create_task():
#     data_size_range = [300000, 500000]  # min and max amount of data in bit
#     cpu_cycle_1bit = 10435  # cpu cycle required for 1 bit of data
#     delay_tolerance = 3  # s
#     data_size_choice = [300000, 350000, 400000, 450000, 500000]
#
#     data_size = np.random.choice(data_size_choice)
#     # data_size = np.random.randint(data_size_range[0], data_size_range[1])
#
#     task = {
#         "data": data_size,
#         "cpu_cycle": cpu_cycle_1bit * data_size,
#         "delay_tolerance": delay_tolerance
#     }
#     return task


def get_fixed_task():
    size = 300000
    task = {
        "data": 4096000,
        "cpu_cycle": 3000e6,
        "dt": 8
    }
    return task


if __name__ == '__main__':
    # task = create_task()
    # print(task["cpu_cycle"] / 3.6)
    for i in range(20):
        print(get_random_task())
