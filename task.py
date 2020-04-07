import numpy as np

# [name, cpu cycle for 1 bit, min data, maximum data]
# applications = [
#     [0, 10435, 80000, 800000, 5],  # speech recognition
#     [1, 25346, 300, 1400, 0.5],  # natural language processing
#     [2, 45043, 300000, 30000000, 300],  # face recognition
#     [3, 34252, 300, 4000, 0.1],  # language translation
#     [4, 54633, 100000, 3000000, 50],  # 3d game processing
#     [5, 40305, 100000, 3000000, 40],  # virtual reality
#     [6, 34532, 100000, 3000000, 40],  # augmented reality
# ]
applications = [
    [0, 10435, 80000, 800000, 8],  # speech recognition
    [1, 25346, 300, 1400, 1.5],  # natural language processing
    [2, 45043, 300000, 30000000, 500],  # face recognition
    [3, 34252, 300, 4000, 1],  # language translation
    [4, 54633, 100000, 3000000, 100],  # 3d game processing
    [5, 40305, 100000, 3000000, 80],  # virtual reality
    [6, 34532, 100000, 3000000, 80],  # augmented reality
]


def get_random_task():
    choiche = np.random.choice(len(applications))
    application = applications[choiche]
    data_size = (application[2] + application[3]) / 2.0
    cpu_cycle = data_size * application[1]
    task = {
        "data": data_size,
        "cpu_cycle": cpu_cycle,
        "dt": application[4]
    }
    return task


def make_task_from_applications(application):
    data_size = (application[2] + application[3]) / 2.0
    cpu_cycle = data_size * application[1]
    task = {
        "data": data_size,
        "cpu_cycle": cpu_cycle,
        "dt": application[4]
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
        "data": 15150000,
        "cpu_cycle": 682401450000,
        "dt": 500
    }
    return task


if __name__ == '__main__':
    # task = create_task()
    # print(task["cpu_cycle"] / 3.6)
    for i in range(20):
        get_random_task()
