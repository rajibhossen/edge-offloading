import csv

import task
from mobile import Mobile
from edge import Edge
from cloud import Cloud

uplink_rate = 7000000 # 10 Mbps
device = Mobile(1)
edge = Edge(uplink_rate, 0.6)
cloud = Cloud(uplink_rate, 0.8)

# profile all the task and save it to a file
# with open("data/app_profiles.csv", "w+") as file:
#     csv_writer = csv.writer(file, delimiter=",")
#     headers = ["applications", "m_time", "m_energy", "e_time", "e_energy", "c_time", "c_energy"]
#     csv_writer.writerow(headers)
#     for app in task.applications:
#         job = task.make_task_from_applications(app)
#         m_total, m_time, m_energy = device.calculate_total_cost(job, 0.5, 0.7)
#         e_total, e_time, e_energy = edge.cal_total_cost(job, 0.5, 0.7)
#         c_total, c_time, c_energy = cloud.cal_total_cost(job, 0.5, 0.6)
#         row = [job, m_time, m_energy, e_time, e_energy, c_time, c_energy]
#         csv_writer.writerow(row)
job = task.get_fixed_task()
m_total, m_time, m_energy = device.calculate_total_cost(job, 0.5, 0)
e_total, e_time, e_energy = edge.cal_total_cost(job, 0.5, 0)
c_total, c_time, c_energy = cloud.cal_total_cost(job, 0.5, 0)
e_r = (m_total - e_total) / m_total
c_r = (m_total - c_total) / m_total
row = [job, m_time, m_energy, e_time, e_energy, c_time, c_energy]
print(e_r, c_r)
print(row)
