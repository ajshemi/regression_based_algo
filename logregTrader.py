#
# Modified by Akpojotor Shemi. This script updates the log regression parameters cumulatively

# Python Script
# with log Regression-based Trading Class
# for Oanda
#
# Python for Algorithmic Trading
# (c) Dr. Yves J. Hilpisch
# The Python Quants GmbH
#

#import python and custom libraries
import argparse
import numpy as np
import pandas as pd
from sklearn import linear_model #scikit learn machine learning in python
from tpqoa import tpqoa


class oaOLSTrader(tpqoa):
    def __init__(self, conf_file, lags, instrument, units):
        tpqoa.__init__(self, conf_file)
        self.data = pd.DataFrame()
        self.lags = lags
        self.instrument = instrument
        self.position = 0
        self.units = units
        self.int_length = self.lags + 5
    
    def create_lags(self): #create lags otherwise known as features
        ''' Creates return lags in the resampled DataFrame object. '''
        self.cols = []
        #self.cols.append('returns')
        for lag in range(1, self.lags+1):
            col = 'returns_%d' % lag
            self.resam[col] = self.resam['returns'].shift(lag)
            self.cols.append(col)
    
    #function/method for logistic regression
    def log_regression(self):
        self.create_lags()
        self.resam.dropna(inplace=True)
        mean=self.resam['returns'].mean()
        std=self.resam['returns'].std()
        
        #function to bucketize data
        def buckets(x):
            v=0
            for b in [mean-std,mean-std/2,mean,mean+std/2,mean+std]:
                if x<b:return v
                v+=1
            return v
        
        for col in self.cols:
            self.resam[col]=self.resam[col].apply(lambda x:buckets(x))
        
        #fit and predict using the logistic Regression model
        
        lm=linear_model.LogisticRegression(C=1e4,solver='lbfgs',multi_class='multinomial')
        logR=lm.fit(self.resam[self.cols],np.sign(self.resam['returns']))
        self.resam['position'] = logR.predict(self.resam[self.cols])
        
    #function/method for ordinary least squares regression
    #def ols_regression(self):
    #    self.create_lags()
    #    self.resam.dropna(inplace=True)
    #    self.regr=np.linalg.lstsq(self.resam[self.cols],\
    #                              self.resam['returns'],rcond=None)[0] 
    #    self.resam['position']=np.sign(np.dot(self.resam[self.cols],self.regr))
            
            
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

        
    def on_success(self, time, bid, ask):
        ''' Method called when new data is retrieved. '''
        print('%3d | ' % self.ticks, time, bid, ask)  # print tick data as received

        # collecting the incoming tick data
        self.data = self.data.append(pd.DataFrame({'bid': bid, 'ask': ask,
                                                   'mid': (bid + ask) / 2},
                                                  index=[pd.Timestamp(time)]))

        # resampling the tick data to a homogeneous time interval
        self.resam = self.data.resample('10s', label='right').last().ffill()

        if len(self.resam) > self.int_length:
            self.int_length = len(self.resam)
            self.resam['returns'] = np.log(self.resam['mid'] /
                                           self.resam['mid'].shift(1))
            #self.ols_regression() #ordinary least squares
            self.log_regression()  #logistic regression
            
            if len(self.resam)%5==0:
                print('\ninterval length %d\n' % self.int_length)
                print(self.resam[['bid','ask','mid', 'position','returns']].tail())
                #print('\nupdated reg parameters: \n', self.regr)
            
            
            # from neutral to long
            if self.position == 0 and self.resam['position'].iloc[-2] > 0 and len(self.resam)>11:
                self.create_order(self.instrument, units=self.units)
                self.position = 1

            # from short to long
            elif self.position == -1 and self.resam['position'].iloc[-2] > 0 and len(self.resam)>11:
                self.create_order(self.instrument, units=2 * self.units)
                self.position = 1

            # from neutral to short
            elif self.position == 0 and self.resam['position'].iloc[-2] < 0 and len(self.resam)>11:
                self.create_order(self.instrument, units=-self.units)
                self.position = -1

            # from long to short
            elif self.position == 1 and self.resam['position'].iloc[-2] < 0 and len(self.resam)>11:
                self.create_order(self.instrument, units=-2 * self.units)
                self.position = -1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('lags', help='number of lags to do regression')
    parser.add_argument('instrument', help='instrument to parse data for')
    parser.add_argument('units', help='number of units to trade')
    args = parser.parse_args()
    ols = oaOLSTrader('pyalgo_sample.cfg', int(args.lags), args.instrument, int(args.units))
    ols.stream_data(stop=100)
