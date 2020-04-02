import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_figures(filename, vendor_name):
    # read csv, set datetime as index column with parse true-pandas DatetimeIndex
    vendor = pd.read_csv(filename, index_col=0, parse_dates=True)
    # print(amazon.head(10))
    # print(amazon.dtypes)
    # set index to datetime column
    # amazon = amazon.set_index(['datetime'])
    # print(amazon.head(10))
    # amazon.index = pd.DatetimeIndex(amazon.index)
    per_hour_mean = vendor.groupby(vendor.index.hour).mean()
    if vendor_name == 'amazon':
        print(per_hour_mean)
    # per_hour_mean.plot(y='us-east-1', use_index=True)
    plot = None
    if vendor_name == 'amazon':
        plot = per_hour_mean.reset_index().plot(title="amazon", x='datetime',
                                                y=['us-east-1', 'us-east-2', 'us-west-1', 'us-west-2'])
    elif vendor_name == 'azure':
        plot = per_hour_mean.reset_index().plot(title="azure", x='datetime',
                                                y=['us-central', 'us-east', 'us-east-2', 'us-west', 'us-west-2',
                                                   'us-north-central', 'us-west-central', 'us-south-central'])
    elif vendor_name == 'do':
        plot = per_hour_mean.reset_index().plot(title="digital ocean", x='datetime',
                                                y=['nyc1', 'nyc2', 'nyc3', 'sfo1', 'sfo2'])
        # plt.savefig("do.png")
    elif vendor_name == 'gcp':
        plot = per_hour_mean.reset_index().plot(title="google cloud", x='datetime',
                                                y=['us-central-1', 'us-east-1', 'us-east-4', 'us-west-1', 'us-west-2',
                                                   'us-west-3'])

    figure = plot.get_figure()
    figure.savefig(vendor_name + ".png")
    plt.show()


if __name__ == '__main__':
    filename = 'aws_data.csv'
    vendor = "amazon"
    plot_figures(filename, vendor)

    filename = 'azure_data.csv'
    vendor = "azure"
    plot_figures(filename, vendor)

    filename = 'digital_ocean_data.csv'
    vendor = "do"
    plot_figures(filename, vendor)

    filename = 'gcp_data.csv'
    vendor = "gcp"
    plot_figures(filename, vendor)
