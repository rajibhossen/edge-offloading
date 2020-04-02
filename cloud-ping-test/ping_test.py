import csv
import os
from datetime import datetime
import threading

import requests
from apscheduler.schedulers.blocking import BlockingScheduler

AWS_URLS = {
    "us-east-1": "http://ec2.us-east-1.amazonaws.com/ping"
    # "us-east-2": "http://ec2.us-east-2.amazonaws.com/ping",
    # "us-west-1": "http://ec2.us-west-1.amazonaws.com/ping",
    # "us-west-2": "http://ec2.us-west-2.amazonaws.com/ping"
}

# AWS_FIELDS = ["datetime", "us-east-1", "us-east-2", "us-west-1", "us-west-2"]
AWS_FIELDS = ["datetime", "us-east-1"]

AZURE_URLS = {
    # "us-central": "https://astcentralus.blob.core.windows.net/public/callback.js",
    # "us-east": "https://asteastus.blob.core.windows.net/public/callback.js",
    # "us-east-2": "https://asteastus2.blob.core.windows.net/public/callback.js",
    # "us-west": "https://astwestus.blob.core.windows.net/public/callback.js",
    # "us-west-2": "https://astwestus2.blob.core.windows.net/public/callback.js",
    # "us-north-central": "https://astnorthcentralus.blob.core.windows.net/public/callback.js",
    # "us-west-central": "https://astwestcentralus.blob.core.windows.net/public/callback.js",
    "us-south-central": "https://astsouthcentralus.blob.core.windows.net/public/callback.js"
}
# AZURE_FIELDS = ["datetime", "us-central", "us-east", "us-east-2", "us-west", "us-west-2", "us-north-central",
#                "us-west-central", "us-south-central"]
AZURE_FIELDS = ["datetime", "us-south-central"]

DIGITAL_OCEAN_URLS = {
    "nyc1": "http://speedtest-nyc1.digitalocean.com/",
    "nyc2": "http://speedtest-nyc2.digitalocean.com/",
    "nyc3": "http://speedtest-nyc3.digitalocean.com/",
    "sfo1": "http://speedtest-sfo1.digitalocean.com/",
    "sfo2": "http://speedtest-sfo2.digitalocean.com/",
}

DO_FIELDS = ["datetime", "nyc1", "nyc2", "nyc3", "sfo1", "sfo2"]

GCP_URLS = {
    "us-central-1": "http://35.186.221.153/ping"
    # "us-east-1": "http://104.196.161.21/ping",
    # "us-east-4": "http://35.186.168.152/ping",
    # "us-west-1": "http://104.199.116.74/ping",
    # "us-west-2": "http://35.236.45.25/ping",
    # "us-west-3": "http://34.106.137.137/ping"
}
# GCP_FIELDS = ["datetime", "us-central-1", "us-east-1", "us-east-4", "us-west-1", "us-west-2", "us-west-3"]
GCP_FIELDS = ["datetime", "us-central-1"]

IBM_URLS = {
    "us-central": "https://srdc-uscentral.skytap.com/latency-check",
    # "us-west-1": "https://srdc-uswest1.skytap.com/latency-check",
    # "us-east-1": "https://srdc-useast1.skytap.com/latency-check",
    # "us-east-2": "https://srdc-useast2.skytap.com/latency-check",
}
# IBM_FIELDS = ["us-west-1", "us-central", "us-east-1", "us-east-2"]
IBM_FIELDS = ["datetime", "us-central"]


def data_collector(vendor, filename, urls):
    with open(filename, "a+") as data_file:
        csv_writer = csv.writer(data_file, delimiter=",")

        if os.stat(filename).st_size == 0:
            if vendor == "aws":
                csv_writer.writerow(AWS_FIELDS)
            elif vendor == "azure":
                csv_writer.writerow(AZURE_FIELDS)
            elif vendor == "IBM":
                csv_writer.writerow(IBM_FIELDS)
            elif vendor == "Google":
                csv_writer.writerow(GCP_FIELDS)
            else:
                pass
        print("[%s] - Collecting %s data" % (datetime.now().strftime("%H:%M:%S"), vendor))
        for i in range(20):
            data = [datetime.now()]
            # print("[%s] [%s] - Collecting %s data" % (i, data[0], vendor))
            for key, value in urls.items():
                try:
                    request = requests.get(value)
                    response_time = round(request.elapsed.total_seconds(), 2)
                    response_time *= 1000  # convert to ms
                except requests.exceptions.Timeout:
                    print("Timeout occured")
                    response_time = None
                data.append(response_time)
            csv_writer.writerow(data)


def main():
    filename = "aws_data.csv"
    # data_collector("aws", filename, AWS_URLS)
    aws_thread = threading.Thread(target=data_collector, args=("aws", filename, AWS_URLS))

    filename = "azure_data.csv"
    # data_collector("azure", filename, AZURE_URLS)
    azure_thread = threading.Thread(target=data_collector, args=("azure", filename, AZURE_URLS))

    # filename = "digital_ocean_data.csv"
    # # data_collector("digital ocean", filename, DIGITAL_OCEAN_URLS)
    # do_thread = threading.Thread(target=data_collector, args=("digital ocean", filename, AWS_URLS))

    filename = "gcp_data.csv"
    # data_collector("google cloud", filename, GCP_URLS)
    gcp_thread = threading.Thread(target=data_collector, args=("Google", filename, GCP_URLS))

    filename = "ibm_data.csv"
    ibm_thread = threading.Thread(target=data_collector, args=("IBM", filename, IBM_URLS))

    aws_thread.start()
    azure_thread.start()
    ibm_thread.start()
    gcp_thread.start()

    aws_thread.join()
    azure_thread.join()
    ibm_thread.join()
    gcp_thread.join()
    print("[%s] - DONE" % (datetime.now().strftime("%H:%M:%S")))


if __name__ == '__main__':
    print("[%s] - Starting Data Collection" % (datetime.now().strftime("%H:%M:%S")))
    scheduler = BlockingScheduler()
    scheduler.add_job(main, 'interval', minutes=1)
    scheduler.start()
# main()

