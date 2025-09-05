""" Charging Optimization Model
Developed by Siobhan Powell and adapted by Sonia Martin"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import cvxpy as cvx
import copy


class ChargingData(object):
    ''''Process raw vehicle charging and driving data'''
    def __init__(self, df):
        df.datetime = pd.to_datetime(df.datetime)
        df = self.round_secondsTOminutes(df)
        self.df = df
        self.df.datetime = pd.to_datetime(self.df.datetime)
        #declare charging level powers in kW
        self.l1_rate = 1.3
        self.l2_rate = 6.6
        self.l3_rate = 50.
        self.l4_rate = 150.
        self.battcap = 83.6 # kWh
        self.eta = .1
        
    def define_weeks(self, min_date, max_date):
        ''''Find all Mondays in input date range'''
        self.min_date = min_date
        self.max_date = max_date
        self.mondays = np.sort(self.df.loc[(self.df.datetime.dt.weekday==0)&(self.df.datetime.dt.date >= min_date)&(self.df.datetime.dt.date < max_date)].datetime.dt.date.unique())
        self.sundays = self.mondays + datetime.timedelta(days=6)
        self.num_weeks = len(self.mondays)
        
    def driver_fulldf_byweek(self, vinid, week_number):
        ''''Create the full, minutely dataframe for each driver (by vinid), one week at a time '''
        self.driver_fulldf(vinid, self.mondays[week_number], self.sundays[week_number])
        self.week_number = week_number
        
    def driver_fulldf(self, vinid, date1, date2):
        ''''Take temporally sparse driver data and processes it to output a minutely dataframe that includes SOC, uncontrolled charging, mileage, access, max charging rate, etc.'''
        #copy the sparse data from input date range to tmp
        tmp = self.df.loc[(self.df['VINID']==vinid)&(self.df.datetime.dt.date>=date1)&(self.df.datetime.dt.date<=date2)].copy(deep=True)
        self.data_sample = tmp.copy(deep=True)
        tmp['old_index'] = tmp.index.values
        tmp = tmp.drop_duplicates(subset='datetime', keep='first')
        tmp.index = tmp.datetime
        tmp2 = tmp.resample('1Min').ffill()
        
        #initialize full minutely dataframe
        fulldf = pd.DataFrame({'datetime':pd.date_range(datetime.datetime(date1.year, date1.month, date1.day, 0, 0), 
                                                        datetime.datetime(date2.year, date2.month, date2.day, 23, 59), freq='1min')})

        #copy in mileage, access, max charging rate, and start time
        colset = ['Mileage', 'Access_50m', 'Max_Charging_Rate_kW', 'SessionStart', 'Home']
        inds_tmp1 = fulldf[fulldf['datetime'].isin(tmp2.index[1:])].index
        inds_tmp2 = tmp2.index[1:]
        inds_tmp3 = np.arange(0, fulldf[fulldf['datetime'].isin(tmp2.index)].index.values[0]+1)
        inds_tmp4 = np.arange(fulldf[fulldf['datetime'].isin(tmp2.index)].index.values[-1]+1, len(fulldf))
        for col in colset:
            fulldf.loc[fulldf.index, col] = np.nan
            fulldf.loc[inds_tmp1, col] = tmp2.loc[inds_tmp2, col].values
            fulldf.loc[inds_tmp3, col] = tmp2.loc[tmp2.index[1], col]
            fulldf.loc[inds_tmp4, col] = tmp2.loc[tmp2.index[-1], col]

        #clean and process SOC values from tmp and place them in fulldf
        #process data to eliminate jumps of SOC change = 1%
        for i in range(1, len(tmp.index)-1):
            if tmp.loc[tmp.index[i],'SOC']-tmp.loc[tmp.index[i-1],'SOC'] ==1 and tmp.loc[tmp.index[i],'Status2'] == 'Unplugged':
                tmp.loc[tmp.index[i],'SOC']=tmp.loc[tmp.index[i-1],'SOC']
                tmp.loc[tmp.index[i+1],'SOC-1']=tmp.loc[tmp.index[i],'SOC']

        #initialize SOC with Nan
        fulldf.loc[fulldf.index, 'SOC'] = np.nan
        #fill in known SOC values from dataset
        fulldf.loc[fulldf[fulldf['datetime'].isin(tmp.index[1:])].index, 'SOC'] = tmp.loc[tmp.index[1:], 'SOC'].values

        #identify charge and discharge indices and adjust indices to capture charging behavior of SOC change > 1%
        charge_inds=np.array([])
        end_inds=np.array([])
        for i in range(1, len(tmp.index)):
            if tmp.loc[tmp.index[i],'SOC']-tmp.loc[tmp.index[i],'SOC-1'] >1:
                charge_inds = np.append(charge_inds,tmp.index[i-1])
                end_inds = np.append(end_inds,tmp.index[i])

        #for each charging session, identify any time periods in which car was parked and plugged in, but not charging 
        #i.e., car charged to 100% SOC before it was unplugged. 
        #example: car at 80% plugged in at 8pm and unplugged at 8am. Without processing, this would show a charging rate < L1. 
        #with processing, we set the charging profile to charge at L2 until 100% SOC then at 0 kW for the remaining time.
        for i, ind_chg in enumerate(charge_inds):            
            #if the average charging rate is a different level than the max rate OR if avg charge bin = bin(max_chg)
            if self.bin_rates(tmp.loc[ind_chg, 'Average_Charging_Rate_kW'])<=self.bin_rates(tmp.loc[ind_chg, 'Max_Charging_Rate_kW']) or np.isnan(tmp.loc[ind_chg, 'Average_Charging_Rate_kW']):
                #find charging rate: 
                #if avg rate and max rate are both L3, use avg rate
                #if avg rate and max rate are both l1, use l1
                #if avg rate and max rate are both l2, use l2
                #if avg rate is l2 and max rate is l3, use l2
                #if avg rate is l1 and max rate is l3, use l2
                #if avg rate is l1 and max rate is l2, use l2
                rate = np.fmax(tmp.loc[ind_chg, 'Average_Charging_Rate_kW'], np.minimum(self.bin_rates(tmp.loc[ind_chg, 'Max_Charging_Rate_kW']), self.l2_rate/(1+self.eta)))
                #solve for actual charge time in minutes given above rate
                actual_elapsed_chg_time = np.round(60* (1/100)*self.battcap*(tmp.loc[end_inds[i], 'SOC'] - tmp.loc[ind_chg, 'SOC']) / rate, decimals=2)
                placeholder_time = tmp.loc[ind_chg, 'datetime'] + datetime.timedelta(minutes=actual_elapsed_chg_time)
                SOC_intermediate = (np.floor(actual_elapsed_chg_time) * rate)*100*(1/60)*(1/self.battcap) +  tmp.loc[ind_chg, 'SOC']
                placeholder_time = placeholder_time - datetime.timedelta(seconds=placeholder_time.second) - datetime.timedelta(microseconds=placeholder_time.microsecond) + datetime.timedelta(minutes=1)
                listed_elapsed = datetime.timedelta.total_seconds(tmp.loc[end_inds[i], 'datetime']-tmp.loc[ind_chg, 'datetime'])/60
                #if the calculated charging time is less than unprocessed data vale, add an SOC value to demarcate the actual end of session
                if actual_elapsed_chg_time < listed_elapsed:
                    fulldf.loc[fulldf.datetime==placeholder_time - datetime.timedelta(minutes=1), 'SOC'] = SOC_intermediate
                    fulldf.loc[fulldf.datetime==placeholder_time, 'SOC'] = tmp.loc[end_inds[i], 'SOC']

            
        #linearly interpolate to fill in remaining missing SOC values so dataframe is not sparse             
        fulldf['SOC'] = fulldf['SOC'].interpolate(method='linear')
        
        #data processing - replace NaN's with actual SOC values
        inds1 = fulldf[fulldf['SOC']!=fulldf['SOC']].index
        inds2 = fulldf[fulldf['SOC']==fulldf['SOC']].index
        fulldf.loc[inds1, 'SOC'] = fulldf.loc[inds2, 'SOC'].values[0]

        #fill in correct values for SOC change, access and plugged series, uncontrolled charging power, and prior SOC (SOC-1)
        fulldf.loc[fulldf.index, 'driving_soc_change'] = 0
        fulldf.loc[fulldf.index, 'access_series'] = 0
        fulldf.loc[fulldf.index, 'plugged_series'] = 0
        fulldf.loc[fulldf.index, 'SOC-1'] = np.nan
        inds = fulldf.index
        fulldf.loc[inds[1:], 'SOC-1'] = np.copy(fulldf.loc[inds[:-1], 'SOC'])
        inds = fulldf.loc[(fulldf['SOC']!=np.nan)].index
        inds2 = fulldf.loc[inds, 'SOC'][fulldf.loc[inds, 'SOC'] - fulldf.loc[inds, 'SOC-1']<0].index
        fulldf.loc[inds2, 'driving_soc_change'] = np.copy(fulldf.loc[inds2, 'SOC'] - fulldf.loc[inds2, 'SOC-1'])
        fulldf.loc[fulldf.index, 'Uncontrolled_charging'] = 0
        stops = self.data_sample['Mileage'].value_counts()[self.data_sample['Mileage'].value_counts()>=2].keys()
        for stop in stops:
            inds = fulldf.loc[(fulldf['Mileage']==stop)].index
            if fulldf.loc[inds, 'Access_50m'].sum() > 0:
                fulldf.loc[inds, 'access_series'] = 1
            else:
                fulldf.loc[inds, 'access_series'] = 0
            if fulldf.loc[inds, 'SessionStart'].sum() > 0:
                fulldf.loc[inds, 'plugged_series'] = 1
            else:
                fulldf.loc[inds, 'plugged_series'] = 0
        inds = fulldf.index
        fulldf.loc[inds[:-1], 'Uncontrolled_charging'] = np.maximum(60*(1/100)*self.battcap*(fulldf.loc[inds[1:], 'SOC'].values - fulldf.loc[inds[1:], 'SOC-1'].values), 0)
        fulldf.loc[0, 'SOC-1'] = fulldf.loc[1, 'SOC-1']
        if (fulldf.Max_Charging_Rate_kW < fulldf.Uncontrolled_charging).sum() > 0:
            fulldf.loc[fulldf.index, 'Max_Charging_Rate_kW_old'] = fulldf.loc[fulldf.index, 'Max_Charging_Rate_kW'].values
            fulldf.Max_Charging_Rate_kW = np.copy(np.maximum(fulldf['Max_Charging_Rate_kW'], fulldf['Uncontrolled_charging']))
        self.fulldf = fulldf.loc[:, ['datetime', 'access_series', 'plugged_series', 'SOC', 'driving_soc_change', 'Max_Charging_Rate_kW', 'Mileage', 'SOC-1', 'Uncontrolled_charging', 'Home']]

    
    def bin_rates(self, power):
        '''Round power up to the next highest charging level, factoring in efficiency losses'''
        if power<=self.l1_rate/(1+self.eta):
            return self.l1_rate/(1+self.eta)
        elif power<=self.l2_rate/(1+self.eta):
            return self.l2_rate/(1+self.eta)
        elif power<=self.l3_rate/(1+self.eta):
            return self.l3_rate/(1+self.eta)
        else:
            return self.l4_rate/(1+self.eta)


    def fixed_set_of_rates(self):
        '''Round the maximum charging rate of each session to L1, L2, L3, or L4'''
        self.fulldf.loc[self.fulldf.index, 'Binned_Max_Rate_kW'] = np.nan
        inds = self.fulldf.loc[self.fulldf['Max_Charging_Rate_kW']<=self.l1_rate].index
        self.fulldf.loc[inds, 'Binned_Max_Rate_kW'] = self.l1_rate
        inds = self.fulldf.loc[(self.fulldf['Max_Charging_Rate_kW']>self.l1_rate)&(self.fulldf['Max_Charging_Rate_kW']<=self.l2_rate)].index
        self.fulldf.loc[inds, 'Binned_Max_Rate_kW'] = self.l2_rate
        inds = self.fulldf.loc[(self.fulldf['Max_Charging_Rate_kW']>self.l2_rate)&(self.fulldf['Max_Charging_Rate_kW']<=self.l3_rate)].index
        self.fulldf.loc[inds, 'Binned_Max_Rate_kW'] = self.l3_rate
        inds = self.fulldf.loc[(self.fulldf['Max_Charging_Rate_kW']>self.l3_rate)].index
        self.fulldf.loc[inds, 'Binned_Max_Rate_kW'] = self.l4_rate
        
    def round_secondsTOminutes(self, tmp):
        ''''Manually round from seconds to minute level datetime column, 
        managing the rollover it may cause across hours, days, months, and years'''

        tmp.datetime = pd.to_datetime(tmp.datetime)
        years = tmp.datetime.dt.year
        months = tmp.datetime.dt.month
        days = tmp.datetime.dt.day
        hours = tmp.datetime.dt.hour
        minutes = tmp.datetime.dt.minute
        seconds = tmp.datetime.dt.second
        inds1 = tmp[np.round(seconds/60)==0].index
        inds2 = tmp[np.round(seconds/60)==1].index

        #initialize rounded values dataframe
        newdf = pd.DataFrame({'datetime_rounded': pd.to_datetime(pd.DataFrame({'Year':years.loc[inds1], 
                                            'Month':months.loc[inds1],
                                            'Day':days.loc[inds1],
                                            'Hour':hours.loc[inds1],
                                            'Minute':minutes.loc[inds1],
                                            'Second':np.zeros((len(inds1),))}))})
        if len(inds2) > 0: #some seconds were rounded up to increase the minute count
            inds3 = tmp.loc[(tmp.datetime.dt.minute<59)&(np.round(seconds/60)==1)].index
            inds4 = tmp.loc[(tmp.datetime.dt.minute==59)&(np.round(seconds/60)==1)].index
            newdf2 = pd.DataFrame({'datetime_rounded': pd.to_datetime(pd.DataFrame({'Year':years.loc[inds3], 
                                            'Month':months.loc[inds3],
                                            'Day':days.loc[inds3],
                                            'Hour':hours.loc[inds3],
                                            'Minute':minutes.loc[inds3]+np.ones((len(inds3),)),
                                            'Second':np.zeros((len(inds3),))}))})
            newdf = pd.concat((newdf, newdf2), axis=0)
            if len(inds4) > 0: #some minutes were rounded up to increase the hour count
                inds5 = tmp.loc[(tmp.datetime.dt.hour<23)&(tmp.datetime.dt.minute==59)&(np.round(seconds/60)==1)].index
                inds6 = tmp.loc[(tmp.datetime.dt.hour==23)&(tmp.datetime.dt.minute==59)&(np.round(seconds/60)==1)].index
                newdf3 = pd.DataFrame({'datetime_rounded': pd.to_datetime(pd.DataFrame({'Year':years.loc[inds5], 
                                            'Month':months.loc[inds5],
                                            'Day':days.loc[inds5],
                                            'Hour':hours.loc[inds5]+np.ones((len(inds5),)),
                                            'Minute':np.zeros((len(inds5),)),
                                            'Second':np.zeros((len(inds5),))}))})
                newdf = pd.concat((newdf, newdf3), axis=0)
                if len(inds6) > 0: #some hours were rounded up to increase the day
                    max_days = {1:31, 2:29, 3:31, 4:30, 5:31, 6:30, 7:31, 8:31, 9:30, 10:31, 11:30, 12:31}
                    months_problem = tmp.loc[inds6, 'datetime'].dt.month.unique()
                    for month_here in months_problem:
                        inds7 = tmp.loc[(tmp.datetime.dt.month==month_here)&(tmp.datetime.dt.day<max_days[month_here])&(tmp.datetime.dt.hour==23)&(tmp.datetime.dt.minute==59)&(np.round(seconds/60)==1)].index
                        inds8 = tmp.loc[(tmp.datetime.dt.month==month_here)&(tmp.datetime.dt.day==max_days[month_here])&(tmp.datetime.dt.hour==23)&(tmp.datetime.dt.minute==59)&(np.round(seconds/60)==1)].index
                        newdf4 = pd.DataFrame({'datetime_rounded': pd.to_datetime(pd.DataFrame({'Year':years.loc[inds7], 
                                            'Month':months.loc[inds7],
                                            'Day':days.loc[inds7]+np.ones((len(inds7),)),
                                            'Hour':np.zeros((len(inds7),)),
                                            'Minute':np.zeros((len(inds7),)),
                                            'Second':np.zeros((len(inds7),))}))})
                        newdf = pd.concat((newdf, newdf4), axis=0)
                        if len(inds8) > 0: #some days were rounded up to increase the month
                            if month_here == 12:
                                newdf5 = pd.DataFrame({'datetime_rounded': pd.to_datetime(pd.DataFrame({'Year':years.loc[inds8]+np.ones((len(inds8),)), 
                                            'Month':np.ones((len(inds8),)),
                                            'Day':np.ones((len(inds8),)),
                                            'Hour':np.zeros((len(inds8),)),
                                            'Minute':np.zeros((len(inds8),)),
                                            'Second':np.zeros((len(inds8),))}))})
                                newdf = pd.concat((newdf, newdf5), axis=0)
                            else:
                                newdf5 = pd.DataFrame({'datetime_rounded': pd.to_datetime(pd.DataFrame({'Year':years.loc[inds8], 
                                            'Month':np.ones((len(inds8),)),
                                            'Day':np.ones((len(inds8),)),
                                            'Hour':np.zeros((len(inds8),)),
                                            'Minute':np.zeros((len(inds8),)),
                                            'Second':np.zeros((len(inds8),))}))})
                                newdf = pd.concat((newdf, newdf5), axis=0)
        newdf = newdf.sort_index()
        tmp['datetime'] = newdf['datetime_rounded']
        return tmp
        
class ChargingAutomation(object):
    '''Run charging analysis for multiple drivers'''
    def __init__(self, min_date, max_date, driver_subset=None, df=None, data=None):
        self.min_date = min_date
        self.max_date = max_date
        if data is None:    
            self.df = df
            self.data = ChargingData(self.df.copy(deep=True))
            self.data.define_weeks(min_date=min_date, max_date=max_date)
            print('Number of Weeks: ', self.data.num_weeks)
        else:
            self.df = data.df.copy(deep=True)
            self.data = copy.deepcopy(data)
        
        if driver_subset is None:
            self.driver_subset = list(self.df['VINID'].unique())
        else:
            self.driver_subset = list(driver_subset)
        print('Max VIN: ', len(self.driver_subset))
        self.num_drivers = len(self.driver_subset)
        
        #initialize dataframes for uncontrolled charging and SOC profiles
        self.cols = ['datetime']
        self.cols.extend(self.driver_subset)
        self.uncontrolled_charging_demand = pd.DataFrame(columns=self.cols) #one column per driver, with the driver ID as the column name
        self.uncontrolled_soc = pd.DataFrame(columns=self.cols)
        self.access = pd.DataFrame(columns=self.cols)
        self.driving_soc_change = pd.DataFrame(columns=self.cols)
        self.binned_max_kW = pd.DataFrame(columns=self.cols)
        self.home = pd.DataFrame(columns=self.cols)
        self.controlled_charging_demand = {}
        self.controlled_charging_soc = {}

        self.l1_rate = 1.3
        self.l2_rate = 6.6
        self.l3_rate = 50.
        self.l4_rate = 150.


    def fixed_set_of_rates(self):
        '''Round the maximum charging rate of each session to L1, L2, L3, or L4'''
        self.fulldf.loc[self.fulldf.index, 'Binned_Max_Rate_kW'] = np.nan
        inds = self.fulldf.loc[self.fulldf['Max_Charging_Rate_kW']<=self.l1_rate].index
        self.fulldf.loc[inds, 'Binned_Max_Rate_kW'] = self.l1_rate
        inds = self.fulldf.loc[(self.fulldf['Max_Charging_Rate_kW']>self.l1_rate)&(self.fulldf['Max_Charging_Rate_kW']<=self.l2_rate)].index
        self.fulldf.loc[inds, 'Binned_Max_Rate_kW'] = self.l2_rate
        inds = self.fulldf.loc[(self.fulldf['Max_Charging_Rate_kW']>self.l2_rate)&(self.fulldf['Max_Charging_Rate_kW']<=self.l3_rate)].index
        self.fulldf.loc[inds, 'Binned_Max_Rate_kW'] = self.l3_rate
        inds = self.fulldf.loc[(self.fulldf['Max_Charging_Rate_kW']>self.l3_rate)].index
        self.fulldf.loc[inds, 'Binned_Max_Rate_kW'] = self.l4_rate

    def calculate_uncontrolled_only_oneweek(self, week, driver_subset=None, verbose=True):
        '''Calculate the uncontrolled charging demand for one given week'''
        if driver_subset is None:
            driver_subset = self.driver_subset
        dt = pd.date_range(self.data.mondays[week], self.data.mondays[week]+datetime.timedelta(days=7), freq='min')[:-1]
        self.uncontrolled_soc = pd.concat((self.uncontrolled_soc, pd.DataFrame({'datetime':dt})), axis=0, ignore_index=True, sort=False)
        self.uncontrolled_charging_demand = pd.concat((self.uncontrolled_charging_demand, pd.DataFrame({'datetime':dt})), axis=0, ignore_index=True, sort=False)
        self.access = pd.concat((self.access, pd.DataFrame({'datetime':dt})), axis=0, ignore_index=True, sort=False)
        self.driving_soc_change = pd.concat((self.driving_soc_change, pd.DataFrame({'datetime':dt})), axis=0, ignore_index=True, sort=False)
        self.binned_max_kW = pd.concat((self.binned_max_kW, pd.DataFrame({'datetime':dt})), axis=0, ignore_index=True, sort=False)
        self.home = pd.concat((self.home, pd.DataFrame({'datetime':dt})), axis=0, ignore_index=True, sort=False)

        inds = self.uncontrolled_charging_demand[self.uncontrolled_charging_demand.datetime.isin(dt)].index

        #for each driver, check if there is data for that week; if so, get uncontrolled charging profile
        for driver in driver_subset:
            if verbose:
                print('Week '+str(week)+' Driver '+str(driver))
            if self.check_driver(driver, week):
                self.data.driver_fulldf_byweek(driver, week)
                self.uncontrolled_soc.loc[inds, driver] = self.data.fulldf['SOC'].values
                self.uncontrolled_charging_demand.loc[inds, driver] = self.data.fulldf['Uncontrolled_charging'].values 
                self.access.loc[inds, driver] = self.data.fulldf['access_series'].values
                self.driving_soc_change.loc[inds, driver] = self.data.fulldf['driving_soc_change'].values
                
                self.fulldf = self.data.fulldf
                self.fixed_set_of_rates()
                self.binned_max_kW.loc[inds, driver] = self.fulldf['Binned_Max_Rate_kW'].values
                self.home.loc[inds, driver] = self.fulldf['Home'].values

    def check_driver(self, driver, week):
        '''Check whether the driver has data and charging in this week, otherwise don't need to run optimization'''
        #only use vehicles with at least 10 datapoints and at least one charging event during that week
        subset = self.data.df.loc[(self.data.df['VINID']==driver)&(self.data.df.datetime.dt.date>=self.data.mondays[week])&(self.data.df.datetime.dt.date<=self.data.sundays[week])].copy(deep=True)
        if (len(subset) > 10) and (subset.Access_50m.sum() > 0):        
            return True
        else:
            return False
    
    def run_control_oneweek(self, week, name='mef_varying_allaccess', access_series_name='access_series', objective_type=None, reg=1, dpdf=None, mef_type='varying', aef_type = None, mef_unique=None, force_uncontrolled=False, force_nouncontrolled=False, save_str=None, verbose=True, driver_subset=None, mindate=None, maxdate=None, timestep='15min'):
        '''Run the control for each driver and each week, compile'''
        if driver_subset is None:
            driver_subset = self.driver_subset
        do_uncontrolled=False
        if (len(self.uncontrolled_charging_demand) == 0) or force_uncontrolled:
            if force_nouncontrolled:
                do_uncontrolled=False
            else:
                do_uncontrolled=True
            
        #get time values depending on timestep size
        if timestep == '15min':
            dt = pd.date_range(self.data.mondays[week], self.data.mondays[week]+datetime.timedelta(days=7), freq='15min')[:-1]
        elif timestep == '1h':
            dt = pd.date_range(self.data.mondays[week], self.data.mondays[week]+datetime.timedelta(days=7), freq='1h')[:-1]
        else:
            dt = pd.date_range(self.data.mondays[week], self.data.mondays[week]+datetime.timedelta(days=7), freq='min')[:-1]
        if name not in self.controlled_charging_demand.keys():
            self.controlled_charging_demand[name] = pd.DataFrame(columns=self.cols) # initialize empty df
            self.controlled_charging_soc[name] = pd.DataFrame(columns=self.cols) # initialize empty df
        
        #set up charging demand dataframes
        self.controlled_charging_demand[name] = pd.concat((self.controlled_charging_demand[name], pd.DataFrame({'datetime':dt})), axis=0, ignore_index=True, sort=False)
        self.controlled_charging_soc[name] = pd.concat((self.controlled_charging_soc[name], pd.DataFrame({'datetime':dt})), axis=0, ignore_index=True, sort=False)
        inds = self.controlled_charging_soc[name][self.controlled_charging_soc[name].datetime.isin(dt)].index
        if do_uncontrolled:
            self.uncontrolled_soc = pd.concat((self.uncontrolled_soc, pd.DataFrame({'datetime':dt})), axis=0, ignore_index=True, sort=False)
            self.uncontrolled_charging_demand = pd.concat((self.uncontrolled_charging_demand, pd.DataFrame({'datetime':dt})), axis=0, ignore_index=True, sort=False)

        #for each driver, run optimization if there is data for that week.
        for idx, driver in enumerate(driver_subset):
            if verbose:
                print('Week '+str(week)+' Driver '+str(driver))
            if self.check_driver(driver, week):
                self.data.driver_fulldf_byweek(driver, week)
                self.data.fixed_set_of_rates()
                self.charging = ChargingControl(self.data, self.data.fulldf)
                if mef_unique is None:
                    self.charging.setup_objective(objective_type=objective_type, reg=reg, dpdf=dpdf, mef_type=mef_type, mindate=mindate, maxdate=maxdate, aef_type=aef_type)
                else:
                    col_num = str(int(np.floor(idx / (len(driver_subset) / (len(mef_unique.columns)-1)))))
                    self.charging.setup_objective(objective_type=objective_type, reg=reg, dpdf=dpdf, mef_type=mef_type, mindate=mindate, maxdate=maxdate, mef_unique_signal=mef_unique, col_num=col_num)
                
                #if no charging, set all values to zero
                if self.data.fulldf.Uncontrolled_charging.max() == 0:
                    self.controlled_charging_demand[name].loc[inds, driver] = 0
                    self.controlled_charging_soc[name].loc[inds, driver] = 0

                #else, try to run the optimization. If it fails, set output values to nan
                else:
                    try:
                        self.charging.run_optimization(access_series_name=access_series_name, timestep=timestep)
                    except:
                        self.controlled_charging_demand[name].loc[inds, driver] = np.nan
                        self.controlled_charging_soc[name].loc[inds, driver] = np.nan
                    try:
                        self.controlled_charging_demand[name].loc[inds, driver] = self.charging.charging_schedule.value
                        self.controlled_charging_soc[name].loc[inds, driver] = self.charging.total_soc.value
                    except:
                        self.controlled_charging_demand[name].loc[inds, driver] = np.nan
                        self.controlled_charging_soc[name].loc[inds, driver] = np.nan
                if do_uncontrolled:
                    self.uncontrolled_soc.loc[inds, driver] = self.data.fulldf['SOC'].values
                    self.uncontrolled_charging_demand.loc[inds, driver] = self.data.fulldf['Uncontrolled_charging'].values       

        if save_str is not None:
            #save charging demand values for L1, L2, and L3 charging (can choose to save controlled and uncontrolled demand and SOC)
            #self.controlled_charging_soc[name].to_csv(save_str+'_controlled_soc_'+name+'.csv')
            self.controlled_charging_demand[name].to_csv(save_str+'_controlled_demand_'+name+'.csv')
            #self.uncontrolled_charging_demand.to_csv(save_str+'_uncontrolled_demand_'+name+'.csv')
            #self.uncontrolled_soc.to_csv(save_str+'_uncontrolled_soc_'+name+'.csv')
            
        
        
