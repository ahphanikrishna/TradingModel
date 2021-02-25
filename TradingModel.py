import pandas
import logging
import quandl
import matplotlib.pyplot as plt
plt.style.use('seaborn-pastel')

# loading quandl
QUANDL_KEY = "zSknuWKyh6Cny1ZyvdYi"
quandl.ApiConfig.api_key = QUANDL_KEY


class StockDataLoader(object):
    def __init__(self, stock_id, start_date, end_date):
        self.stock_id = stock_id
        self.start_date = start_date
        self.end_date = end_date

        # generating the data while initializing
        self.stock_data = self.data()

    def data(self):
        """
            :Description: Generate a time series data for a given stock from start date to end date
            :return: pandas dataframe
        """

        logging.info(msg="Fetching data for {0}".format(self.stock_id))

        try:
            df = quandl.get('BSE/BOM' + self.stock_id, start_date=self.start_date, end_date=self.end_date)
            return df
        except Exception as e:
            logging.warning("Unable to fetch data for {0}".format(self.stock_id))
            return pandas.DataFrame()

    def generate_hieken_ashi(self):
        """
            :Description:
                Close= 1/4(Open+High+Low+Close) - (The average price of the current bar)
                Open= 1/2(Open of Prev. Bar+Close of Prev. Bar) - (The midpoint of the previous bar)
                High=Max[High, Open, Close]
                Low=Min[Low, Open, Close]

            :return: Adds hieken-ashi columns to the data
        """

        shift_1d = self.stock_data.shift(periods=1)
        stock_data = self.stock_data
        shift_1d.columns = ["Prev. " + x for x in shift_1d.columns]
        shift_1d = pandas.concat([stock_data, shift_1d], axis=1)

        shift_1d.loc[:, "HK Close"] = shift_1d.loc[:, ['Open', 'High', 'Low', 'Close']].sum(axis=1)/4.0
        shift_1d.loc[:, "HK Open"] = shift_1d.loc[:, ["Prev. Close", 'Prev. Open']].sum(axis=1) / 2.0
        shift_1d.loc[:, "HK High"] = shift_1d.loc[:, ['High', 'Open', 'Close']].max(axis=1)
        shift_1d.loc[:, "HK Low"] = shift_1d.loc[:, ['Low', 'Open', 'Close']].min(axis=1)

        cols = ['Open', 'High', 'Low', 'Close']
        prevcols = ['Prev. {0}'.format(item) for item in cols]
        hkcols = ['HK {0}'.format(item) for item in cols]
        shift_1d = shift_1d[cols + prevcols + hkcols]
        shift_1d.loc[:, "Candle Type"] = shift_1d.loc[:, hkcols].apply(candle_type, axis=1)

        return shift_1d

    @staticmethod
    def year_returns(data, exact=True):
        if exact:
            shift_1year = data["Open"].copy()
            shift_1year.index = shift_1year.index.map(lambda x: x.replace(year=x.year+1))
            shift_1year = shift_1year.rename("Prev Year Open")
            shift_1year = shift_1year[shift_1year.index.isin(data.index)]
            data = pandas.concat([data, shift_1year], axis=1)
            data.loc[:, "Yearly Return"] = (data["Open"] - data["Prev Year Open"]) / data["Prev Year Open"]
        else:
            data.loc[:, "Yearly Return"] = (data["Open"] - data["Open"].shift(periods=252)) / data["Open"].shift(
                periods=252)
        return data

    @staticmethod
    def stock_analysis(data):
        """
        :param data: Hiken Aishi data with categories
        :return: pandas dataframe with entry and exit points
        """
        cols = ['Open', 'High', 'Low', 'Close', 'Yearly Return', 'Candle Type', 'Prev Candle Type']
        prev_doji = data['Candle Type'].shift(periods=1)
        prev_doji = prev_doji.rename("Prev Candle Type")
        data = pandas.concat([data, prev_doji], axis=1)
        data.loc[:, "Algo-Stats"] = data.loc[:, cols].apply(stock_entry_exit, axis=1)

        entry_data = data.loc[data["Algo-Stats"] == 'ENTRY', ["Open"]]
        entry_data.loc[:, "Entry Date"] = entry_data.index
        exit_data = data.loc[data["Algo-Stats"] == 'EXIT', ["Open"]]
        exit_data.loc[:, "Exit Date"] = exit_data.index

        df = pandas.concat([entry_data, exit_data], axis=1)
        df.iloc[:, -2:] = df.iloc[:, -2:].bfill(axis='rows')
        df = df[~df['Entry Date'].isnull()]

        df.columns = ['Entry Open', 'Entry Date', 'Exit Open', 'Exit Date']
        df.loc[:, "Returns"] = df.loc[:, 'Exit Open'] - df.loc[:, 'Entry Open']

        msg = "With the investment of {:.2f} the returns made is {:.2f} and final amount available is {:.2f}"
        logging.warning(msg.format(float(df.iloc[:, 0].sum()), float(df.iloc[:, -1].sum()), float(df.iloc[:, -3].sum())))

        return df

    def plot_data(self):
        data = self.stock_data
        plt.figure()
        width = 1
        width2 = 0.1
        dataup = data[data.Close >= data.Open]
        datadown = data[data.Close < data.Open]

        plt.bar(dataup.index, dataup.Close - dataup.Open, width, bottom=dataup.Open, color='g')
        plt.bar(dataup.index, dataup.High - dataup.Close, width2, bottom=dataup.Close, color='g')
        plt.bar(dataup.index, dataup.Low - dataup.Open, width2, bottom=dataup.Open, color='g')

        plt.bar(datadown.index, datadown.Close - datadown.Open, width, bottom=datadown.Open, color='r')
        plt.bar(datadown.index, datadown.High - datadown.Open, width2, bottom=datadown.Open, color='r')
        plt.bar(datadown.index, datadown.Low - datadown.Close, width2, bottom=datadown.Close, color='r')
        plt.grid()
        plt.show()


def candle_type(row):
    open1 = row[0]
    close = row[3]
    high = row[1]
    low = row[2]
    category = ""

    if (open1 <= low) and (close > open1) and (high >= close):
        category = "BULLISH"
    elif (open1 >= high) and (close < open1) and (low <= close):
        category = "BEARISH"
    elif (high >= open1 and high >= close) and (low <= open1 and low <= close):
        category = "DOJI"

    return category


def stock_entry_exit(row):
    algo_stats = ""
    if row['Yearly Return'] > 0.1:
        if row['Prev Candle Type'] == "DOJI" and row['Candle Type'] == "BULLISH":
            algo_stats = "ENTRY"
        elif (row["Candle Type"] == "BEARISH") and (row['Prev Candle Type'] == "DOJI" or row[
                'Prev Candle Type'] == "BULLISH"):
            algo_stats = "EXIT"
    return algo_stats


if __name__ == '__main__':
    stock_id = "500012"
    start_date = "01-01-2019"
    end_date = "31-12-2020"

    loader = StockDataLoader(stock_id, start_date, end_date)
    # loader.plot_data()
    hkdf = loader.generate_hieken_ashi()
    hkdf = loader.year_returns(hkdf, False)
    df = loader.stock_analysis(hkdf)
    print(df)
