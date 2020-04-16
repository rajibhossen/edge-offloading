import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_figures(filename, vendor_name):
    # read csv, set datetime as index column with parse true-pandas DatetimeIndex
    # vendor = pd.read_csv(filename, index_col=0, parse_dates=True)
    vendor = pd.read_csv(filename)
    print(vendor_name)
    print(vendor.describe())
    times = pd.DatetimeIndex(vendor.datetime)
    # print(times)
    fig, ax = plt.subplots(figsize=(15, 7))
    vendor = vendor.groupby([times.date, times.hour]).mean()
    vendor.index.names = ['date', 'hour']
    # print(vendor.index)
    # vendor = vendor.MultiIndex.set_names('date', level=0)
    # vendor = vendor.MultiIndex.set_names('hour', level=1)
    # print(vendor)
    # vendor = vendor.resample('D', on=vendor.index).mean()
    # indexes = vendor.index.floor('D')
    # print(indexes)

    # print(amazon.head(10))
    # print(amazon.dtypes)
    # set index to datetime column
    # amazon = amazon.set_index(['datetime'])
    # print(amazon.head(10))
    # amazon.index = pd.DatetimeIndex(amazon.index)
    # per_day = vendor.groupby(vendor.index.date).mean()
    # per_hour_mean = vendor.groupby(vendor.index.date).mean()
    # per_day = vendor.groupby(vendor.index.date)
    # for row in per_day:
    #     print(row)
    #
    # # if vendor_name == 'amazon':
    # #     print(per_day)
    # # per_hour_mean.plot(y='us-east-1', use_index=True)
    # print(vendor.index.get_level_values(0))
    # print(vendor.loc[('2020-03-31',),])
    # for date, new_df in vendor.groupby(level=0):
    #     plot = new_df.plot(x=new_df.index, y='us-east-1')
    #     break
    # for col in vendor:
    #     print(col)

    plot = None
    if vendor_name == 'amazon':
        # y_axis = ['us-east-1', 'us-east-2', 'us-west-1', 'us-west-2']
        y_axis = ['us-east-1']
        # plot = vendor.reset_index().plot(title="amazon", x='datetime', y=y_axis)
        plot = vendor.plot(title="amazon", ax=ax)
    elif vendor_name == 'azure':
        # y_axis = ['us-central', 'us-east', 'us-east-2', 'us-west', 'us-west-2', 'us-north-central',
        # 'us-west-central', 'us-south-central']
        y_axis = ['us-south-central']
        # plot = vendor.reset_index().plot(title="azure", x='datetime', y=y_axis)
        plot = vendor.plot(title="azure", ax=ax)
    elif vendor_name == 'ibm':
        y_axis = ['us-central']
        # plot = vendor.reset_index().plot(title="IBM Cloud", x='datetime', y=y_axis)
        plot = vendor.plot(title="ibm", ax=ax)
        # plt.savefig("do.png")
    elif vendor_name == 'gcp':
        y_axis = ['us-central-1']
        # plot = vendor.reset_index().plot(title="google cloud", x='datetime', y=['us-central-1'])
        plot = vendor.plot(title="google cloud", ax=ax)

    figure = plot.get_figure()
    figure.savefig(vendor_name + "_1wk_trace.png")
    plt.show()


if __name__ == '__main__':
    filename = '1wk_trace/aws_data.csv'
    vendor = "amazon"
    plot_figures(filename, vendor)

    filename = '1wk_trace/azure_data.csv'
    vendor = "azure"
    plot_figures(filename, vendor)

    filename = '1wk_trace/ibm_data.csv'
    vendor = "ibm"
    plot_figures(filename, vendor)

    filename = '1wk_trace/gcp_data.csv'
    vendor = "gcp"
    plot_figures(filename, vendor)
