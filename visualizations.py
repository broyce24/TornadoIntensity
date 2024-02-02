import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point


def plot_tornadoes(data, country):
    """
  Post: This functions graphs the tornadoes from our dataset. There are 6
        different plots. The first plot (top left) plots all of the tornadoes
        with a legend indicating their intensity. The other plots show
        represent the intensity of various different tornadoes.
  """
    # Filters to determine contiguous US
    not_virgin = data['st'] != 'VI'
    not_alaska = data['st'] != 'AK'
    not_hawaii = data['st'] != 'HI'
    not_puerto = data['st'] != 'PR'

    # Drop 0 coordinates (described in dataset column labels document)
    not_0_lat = data['slat'] != 0
    not_0_lon = data['slon'] != 0

    # Limits data to contiguous US
    data = data.loc[not_virgin & not_alaska & not_hawaii & not_puerto
                    & not_0_lat & not_0_lon]

    # Limit data to contiguous US
    country = country[(country['NAME'] != 'Alaska')
                      & (country['NAME'] != 'Hawaii')
                      & (country['NAME'] != 'Puerto Rico')]

    # Creates Point objects from the other lat/lon in dataset
    coordinates = zip(data['slon'], data['slat'])
    data['coordinates'] = [
        Point(lon, lat) for lon, lat in coordinates
    ]

    # Creates GeoDataFrame
    tornadoes = gpd.GeoDataFrame(data, geometry='coordinates')

    # Creates plot and axs
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(30, 20),
                            facecolor='coral')

    # Plots overall (top left)
    country.plot(ax=axs[0, 0], color='grey')
    tornadoes.plot(ax=axs[0, 0], column='mag', legend=True, markersize=10)
    axs[0, 0].set_title('All tornadoes in the contiguous United States')

    # Plots f1 (top right)
    country.plot(ax=axs[0, 1], color='grey')
    tornadoes[tornadoes['mag'] == 1].plot(ax=axs[0, 1], column='mag',
                                          markersize=10)
    axs[0, 1].set_title('All F1 tornadoes in the contiguous United States')

    country.plot(ax=axs[1, 0], color='grey')
    tornadoes[tornadoes['mag'] == 2].plot(ax=axs[1, 0], column='mag',
                                          markersize=10)
    axs[1, 0].set_title('All F2 tornadoes in the contiguous United States')

    country.plot(ax=axs[1, 1], color='grey')
    tornadoes[tornadoes['mag'] == 3].plot(ax=axs[1, 1], column='mag',
                                          markersize=10)
    axs[1, 1].set_title('All F3 tornadoes in the contiguous United States')

    country.plot(ax=axs[2, 0], color='grey')
    tornadoes[tornadoes['mag'] == 4].plot(ax=axs[2, 0], column='mag',
                                          markersize=10)
    axs[2, 0].set_title('All F4 tornadoes in the contiguous United States')

    country.plot(ax=axs[2, 1], color='grey')
    tornadoes[tornadoes['mag'] == 5].plot(ax=axs[2, 1], column='mag',
                                          markersize=10)
    axs[2, 1].set_title('All F5 tornadoes in the contiguous United States')


def plot_tornado_season(data):
    """
  Post: This function plots the number of tornadoes that have occured in each
.DS_Store
  """
    # Labels for graph
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep',
              'Oct', 'Nov', 'Dec']
    total_by_month = data['mo'].value_counts().sort_index()
    fig2, ax = plt.subplots(1, figsize=(30, 20))

    ax.bar(months, total_by_month)
    ax.set_title("Total number of tornadoes in each month (1950-2021)")


def main():
    country = gpd.read_file('/content/drive/MyDrive/Tornado/\
                          gz_2010_us_040_00_5m.json')
    data = pd.read_csv('/content/drive/MyDrive/Tornado/\
                     1950-2021_all_tornadoes.csv')
    plot_tornadoes(data, country)
    plot_tornado_season(data)


if __name__ == '__main__':
    main()
