#
# Python Script
# with OLS Regression-based Trading Class
# for Oanda
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#
import argparse
import numpy as np
import pandas as pd
from tpqoa import tpqoa

# parameters for signal generation ("secret sauce")
reg = np.array([-0.02833527, -0.00774105, -0.00801869,  0.01874015,  0.00698399])


class oaOLSTrader(tpqoa):
    def __init__(self, conf_file, reg, instrument, units):
        tpqoa.__init__(self, conf_file)
        self.data = pd.DataFrame()
        self.lags = len(reg)
        self.reg = reg
        self.instrument = instrument
        self.position = 0
        self.units = units
        self.int_length = self.lags + 1

    def stream_data(self, stop=None):
        ''' Starts a real-time data stream.

        Parameters
        ==========
        instrument: string
            valid instrument name
        '''
        self.ticks = 0
        response = self.ctx_stream.pricing.stream(
            self.account_id, snapshot=True,
            instruments=self.instrument)
        for msg_type, msg in response.parts():
            if msg_type == 'pricing.Price':
                self.ticks += 1
                self.on_success(msg.time,
                                float(msg.bids[0].price),
                                float(msg.asks[0].price))
                if stop is not None:
                    if self.int_length >= stop:
                        # closing out long position
                        if self.position == 1:
                            self.create_order(self.instrument,
                                              units=-self.units)
                        # closing out short position
                        elif self.position == -1:
                            self.create_order(self.instrument,
                                              units=self.units)
                        break

    def create_lags(self):
        ''' Creates return lags in the resampled DataFrame object. '''
        self.cols = []
        self.cols.append('returns')
        for lag in range(1, self.lags):
            col = 'returns_%d' % lag
            self.resam[col] = self.resam['returns'].shift(lag)
            self.cols.append(col)

    def on_success(self, time, bid, ask):
        ''' Method called when new data is retrieved. '''
        print('%3d | ' % self.ticks, time, bid, ask)

        # collecting the incoming tick data
        self.data = self.data.append(pd.DataFrame({'bid': bid, 'ask': ask,
                                                   'mid': (bid + ask) / 2},
                                                  index=[pd.Timestamp(time)]))

        # resampling the tick data to a homogeneous time interval
        self.resam = self.data.resample('1min', label='right').last().ffill()

        if len(self.resam) > self.int_length:
            self.int_length = len(self.resam)
            self.resam['returns'] = np.log(self.resam['mid'] /
                                           self.resam['mid'].shift(1))
            self.create_lags()

            # deriving position forecast
            self.resam.dropna(inplace=True)
            self.resam['position'] = np.sign(np.dot(self.resam[self.cols],
                                                    self.reg))

            print('\ninterval length %d\n' % self.int_length)
            print(self.resam[['mid', 'position']].tail())

            # from neutral to long
            if self.position == 0 and self.resam['position'].iloc[-2] > 0:
                self.create_order(self.instrument, units=self.units)
                self.position = 1

            # from short to long
            elif self.position == -1 and self.resam['position'].iloc[-2] > 0:
                self.create_order(self.instrument, units=2 * self.units)
                self.position = 1

            # from neutral to short
            elif self.position == 0 and self.resam['position'].iloc[-2] < 0:
                self.create_order(self.instrument, units=-self.units)
                self.position = -1

            # from long to short
            elif self.position == 1 and self.resam['position'].iloc[-2] < 0:
                self.create_order(self.instrument, units=-2 * self.units)
                self.position = -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('instrument', help='instrument to parse data for')
    parser.add_argument('units', help='number of units to trade')
    args = parser.parse_args()
    ols = oaOLSTrader('pyalgo_sample.cfg', reg, args.instrument, int(args.units))
    ols.stream_data(stop=200)