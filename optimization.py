import cvxpy as cp
import numpy as np
import pandas as pd
import json
import datetime
import os

class v2gOptimization(object):
    def __init__(self, params_file, batt_params_file):
        """
        Initializes the optimization object with parameters from the input files.

        Args:
        params_file: string with parameters file locations
        batt_params_file: string with battery model parameters file location
        """ 

        params = json.load(open(params_file))
        batt_params = json.load(open(batt_params_file))

        self.l1_rate = params['l1_rate']
        self.l2_rate = params['l2_rate']
        self.l3_rate = params['l3_rate']
        self.l4_rate = params['l4_rate']
        self.battcap = params['batt_cap']
        self.batt_cost = params['batt_cost']
        self.min_date_price = params['min_date_sim']
        self.max_date_price = params['max_date_sim']
        self.min_date_vehicle = params['min_date_data']
        self.max_date_vehicle = params['max_date_data']
        self.timestep_opt = 1/ params['timestep_per_hr_opt']
        self.region = params['region']
        self.eta = params['eta_batt']
        self.v2g_max = params['v2g_max']
        self.num_trials = params['num_trials']
        self.results_path = os.getcwd() + params['results_path']
        self.data_path = os.getcwd() + params['data_path']

        #set up hourly prices
        self.hourly_price_type = params['hourly_price_type']
        self.price_file = params['price_file']
        self.prices = pd.read_csv(self.price_file, header=0)
        if self.hourly_price_type == 'pge_dynamic':
            self.prices['datetime'] = pd.to_datetime(self.prices['Date'] + ' ' + self.prices['Time (PT) '], format='mixed')
        else:
            print('PG&E hourly prices not set')
        
        #set up ELRP demand response program
        self.elrp = params['elrp']
        if self.elrp:
            self.elrp_seed = params['elrp_seed']
            self.elrp_total_num_days = params['elrp_total_num_days']
            self.elrp_compensation = params['elrp_compensation']
        
        #set up TOU prices
        self.tou = params['tou']
        if self.tou == 'EV2A':
            self.ev2a_summer_file = params['ev2a_summer_file']
            self.ev2a_summer = pd.read_csv(self.ev2a_summer_file, header=None)
            self.ev2a_winter_file = params['ev2a_winter_file']
            self.ev2a_winter = pd.read_csv(self.ev2a_winter_file, header=None)
        else:
            print('TOU prices not set')

        #set public charging price
        self.public_chg_price = params['public_charging_price']

        #set battery model parameters
        self.a = batt_params['a']
        self.b = batt_params['b']
        self.c = batt_params['c']
        self.d = batt_params['d']
        self.e = batt_params['e']
        self.Ea = batt_params['Ea']
        self.R = batt_params['R']
        self.alpha = batt_params['alpha']
        self.beta = batt_params['beta']
        self.gamma = batt_params['gamma']
        self.delta = batt_params['delta']
        self.eol = batt_params['eol']
        self.nom_voltage = batt_params['nom_voltage']
        self.parallel = batt_params['parallel']
        self.series = batt_params['series']
        
    def define_weeks(self, min_date, max_date, data):
        ''''
        Finds all Mondays and number of weeks in input date range.
        Args:
            min_date: a Datetime object
            max_date: a Datetime object
            data: a pandas dataframe with a datetime column

        Returns: a list of all Mondays in the input date range and the number of weeks in the input date range
        '''

        min_date = datetime.datetime.strptime(min_date, '%Y-%m-%d')
        min_date = min_date.date()
        max_date = datetime.datetime.strptime(max_date, '%Y-%m-%d')
        max_date = max_date.date()
        
        mondays = np.sort(data.loc[(data.datetime.dt.weekday==0)&(data.datetime.dt.date >= min_date)&(data.datetime.dt.date < max_date)].datetime.dt.date.unique())
        num_weeks = len(mondays)
        return mondays, num_weeks


    def bin_rates(self, power):
        '''Round power up to the next highest charging level, factoring in efficiency losses
        Args: 
            power: float 
        
        Returns: float of binned power
        '''
        if power<=self.l1_rate * self.eta:
            return self.l1_rate * self.eta
        elif power<=self.l2_rate * self.eta:
            return self.l2_rate * self.eta
        elif power<=self.l3_rate * self.eta:
            return self.l3_rate * self.eta
        else:
            return self.l4_rate * self.eta
        

    def call_optimization(self, vin):
        """
        Runs 1) uncontrolled pricing, 2) V1G managed pricing, and 3) V2G Home, and 4) V2G Everywhere for all price weeks for the input vin
        
        Args:
        vin: particular vin ID to run optimization for
        """ 

        #define cvxpy problems with parameters
        self.num_timesteps = int(24*7 / self.timestep_opt)
        self.data_week_driving_soc_change = cp.Parameter(self.num_timesteps)
        self.combined_access = cp.Parameter(self.num_timesteps)
        self.price_week = cp.Parameter(self.num_timesteps)
        self.price_week_elrp = cp.Parameter(self.num_timesteps)
        self.public_chg_price_week = cp.Parameter(self.num_timesteps)
        self.tou_price_week = cp.Parameter(self.num_timesteps)
        self.uncontrolled_power_home = cp.Parameter(self.num_timesteps, nonneg=True)
        self.uncontrolled_power_away = cp.Parameter(self.num_timesteps, nonneg=True)
        self.max_power_away = cp.Parameter(self.num_timesteps, nonneg=True)
        self.max_power_home = cp.Parameter(self.num_timesteps, nonneg=True)
        self.access = cp.Parameter(self.num_timesteps, nonneg=True)
        self.home = cp.Parameter(self.num_timesteps, nonneg=True)
        self.soc_init_v2g_home = cp.Parameter(nonneg=True)
        self.soc_init_v2g_everywhere = cp.Parameter(nonneg=True)

        self.calendar_aging_coeffs()
        self.run_optimization_managed()
        self.run_optimization_v2g_home()
        self.run_optimization_v2g_everywhere()
        
        data_uncontrolled = pd.read_csv(self.data_path +'/all_uncontrolled_demand_individualdrivers_'+self.min_date_vehicle + '_to_' + self.max_date_vehicle + '.csv')
        data_uncontrolled.datetime = pd.to_datetime(data_uncontrolled.datetime)
        mondays, num_weeks = self.define_weeks(self.min_date_vehicle, self.max_date_vehicle, data_uncontrolled)
        del data_uncontrolled
        sundays = mondays + datetime.timedelta(days=6)

        mondays_price, num_weeks = self.define_weeks(self.min_date_price, self.max_date_price, self.prices)
        cost_uncontrolled = np.full((num_weeks, ), np.nan)
        cost_managed = np.full((num_weeks,), np.nan)
        cost_v2g_home = np.full((num_weeks,), np.nan)
        cost_v2g_everywhere = np.full((num_weeks,), np.nan) 

        print('VIN:', vin)
        #find which weeks have data for this particular vin and create week num vector
        uncontrolled_data = pd.read_csv(self.data_path +'/all_uncontrolled_demand_individualdrivers_'+self.min_date_vehicle + '_to_' + self.max_date_vehicle +'.csv')
        uncontrolled_data.datetime = pd.to_datetime(uncontrolled_data.datetime)
        valid_vin_weeks = []
        for week_vehicle, monday in enumerate(mondays):
            monday_midnight = datetime.datetime.combine(monday, datetime.time.min)
            idx = np.where(uncontrolled_data.datetime == monday_midnight)[0][0]
            if np.sum(uncontrolled_data.iloc[idx:idx+24*7*60,int(vin)+1], axis=0)>20*60:
                valid_vin_weeks.append(week_vehicle)
        del uncontrolled_data

        #repeat vin-weeks until this vin has data for every week
        vin_weeks_vector = np.array(valid_vin_weeks)
        vin_weeks_vector = np.tile(vin_weeks_vector, 60)
        vin_weeks_vector = vin_weeks_vector[:num_weeks]

        driver = int(vin)
        
        #loop through each price_week
        for week_price, start in enumerate(mondays_price):
            week = vin_weeks_vector[week_price]
            print('Price Week:', week_price + 1, 'of', num_weeks)

            #get the dynamic price vector for the price week
            price_week = self.prices.loc[(self.prices.datetime.dt.date >= start)&(self.prices.datetime.dt.date <= start + datetime.timedelta(days=6))].Price.values
            #we want the prices for every minute, so need to upsample
            self.price_week.value = np.repeat(price_week / 100, 60)

            if self.elrp:
                #set up the ELRP price vector for the price week
                tmp_elrp = self.price_week.value
                #pick 10 random days to be ELRP days
                np.random.seed(self.elrp_seed)
                elrp_days = np.random.choice(np.arange(121,self.elrp_total_num_days+121), 10, replace=False)
                elrp_start_times = np.random.choice([16, 17, 18], 10, replace=True)
                #loop through each day to check if it's an ELRP day
                for i in range(7):
                    day = start + datetime.timedelta(days=i)
                    if day.timetuple().tm_yday in elrp_days:
                        start_time = i*24*60 + elrp_start_times[np.where(elrp_days==day.timetuple().tm_yday)[0][0]]*60
                        tmp_elrp[start_time:start_time + 3 * 60 ] = self.elrp_compensation
                self.price_week_elrp.value = tmp_elrp

            
            #get the fixed TOU price vector for the price week
            if self.tou == 'EV2A':
                tmp = np.zeros((24*7,))
                for i in range(7):
                    day = start + datetime.timedelta(days=i)
                    if day.month >= 6 and day.month <= 9:
                        tmp[i*24:(i+1)*24] = self.ev2a_summer.iloc[:,0].values
                    else:
                        tmp[i*24:(i+1)*24] = self.ev2a_winter.iloc[:,0].values
                self.tou_price_week.value = np.repeat(tmp, 60)

            #set the public charging price vector for the price week
            self.public_chg_price_week.value = self.public_chg_price * np.ones((self.num_timesteps,))

            data_uncontrolled = pd.read_csv(self.data_path + '/all_uncontrolled_demand_individualdrivers_'+self.min_date_vehicle + '_to_' + self.max_date_vehicle + '.csv')
            data_uncontrolled.datetime = pd.to_datetime(data_uncontrolled.datetime)
            datetime_vec =data_uncontrolled.datetime
            data_week_uncontrolled = data_uncontrolled.loc[(datetime_vec.dt.date >= mondays[week])&(datetime_vec.dt.date <= sundays[week]), str(driver)]
            del data_uncontrolled

            data_access = pd.read_csv(self.data_path +'/access_individualdrivers_'+self.min_date_vehicle + '_to_' + self.max_date_vehicle + '.csv')
            data_week_access = data_access.loc[(datetime_vec.dt.date >= mondays[week])&(datetime_vec.dt.date <= sundays[week]), str(driver)]
            del data_access

            data_driving_soc_change = pd.read_csv(self.data_path + '/driving_soc_change_'+self.min_date_vehicle + '_to_' + self.max_date_vehicle + '.csv')
            data_week_driving_soc_change = data_driving_soc_change.loc[(datetime_vec.dt.date >= mondays[week])&(datetime_vec.dt.date <= sundays[week]), str(driver)]
            del data_driving_soc_change

            data_binned_max_kW = pd.read_csv(self.data_path + '/binned_max_kW_'+self.min_date_vehicle + '_to_' + self.max_date_vehicle + '.csv')
            data_week_binned_max_kW = data_binned_max_kW.loc[(datetime_vec.dt.date >= mondays[week])&(datetime_vec.dt.date <= sundays[week]), str(driver)]
            del data_binned_max_kW

            data_home = pd.read_csv(self.data_path +'/home_'+self.min_date_vehicle + '_to_' + self.max_date_vehicle + '.csv')
            data_week_home = data_home.loc[(datetime_vec.dt.date >= mondays[week])&(datetime_vec.dt.date <= sundays[week]), str(driver)]
            del data_home

            data_soc = pd.read_csv(self.data_path +'/uncontrolled_soc_'+self.min_date_vehicle + '_to_' + self.max_date_vehicle + '.csv')
            data_week_soc = data_soc.loc[(datetime_vec.dt.date >= mondays[week])&(datetime_vec.dt.date <= sundays[week]), str(driver)]
            del data_soc
            
            self.uncontrolled_home = data_week_home
            self.uncontrolled_away  = data_week_access.astype(bool) & ~data_week_home.astype(bool)
            self.access.value = np.array(data_week_access.astype(bool))
            self.home.value = np.array(data_week_home.astype(bool))

            #set up necessary charging combinations:
            self.uncontrolled_power_away.value = np.array(data_week_uncontrolled * self.uncontrolled_away)
            self.uncontrolled_power_home.value = np.array(data_week_uncontrolled * self.uncontrolled_home)

            #save if necessary
            power_u = np.copy(self.uncontrolled_power_home.value + self.uncontrolled_power_away.value)
            power_u_drive = np.copy(data_week_driving_soc_change)
            power_u_drive[power_u_drive>=0] = 0
            power_u_drive[power_u_drive<0] = power_u_drive[power_u_drive<0] / 100 *self.battcap / self.timestep_opt

            self.uncontrolled_soc = data_week_soc
            cyclic_u, calendar_u, batt_cost_u = self.batt_cost_vec(power_u + power_u_drive, self.uncontrolled_soc)
            
            batt_cost_u = (cyclic_u.value + calendar_u) * (self.batt_cost * self.battcap) / self.eol
            cost_uncontrolled[week_price] = self.baseline_cost(self.uncontrolled_power_away.value, self.public_chg_price_week.value) + self.baseline_cost(self.uncontrolled_power_home.value, self.tou_price_week.value)  + np.sum(batt_cost_u.value)

            data_week_binned_max_kW[data_week_binned_max_kW.isna()] = 0

            self.data_week_driving_soc_change.value = np.array(data_week_driving_soc_change)
            self.max_power_away.value = np.array(data_week_binned_max_kW * (np.array(data_week_access) & ~data_week_home.astype(bool)))
            self.max_power_home.value = np.array(data_week_binned_max_kW * np.array(data_week_home))
            
            del data_week_binned_max_kW
            del data_week_access
            del data_week_home
            del data_week_soc
            del data_week_uncontrolled
            del data_week_driving_soc_change

            self.prob_managed.solve(solver='MOSEK', ignore_dpp=True)
            
            try:
                cost_managed[week_price] = self.charging_cost_managed.value  + np.sum(self.batt_deg_cost_managed.value) 
            except:
                cost_managed[week_price] = np.nan
            
            self.prob_v2g.solve(solver='MOSEK', ignore_dpp=True, verbose=False)
            
            try:
                cost_v2g_home[week_price] = self.obj_v2g.value #- self.batt_deg_cost_v2g.value
            except:
                cost_v2g_home[week_price] = np.nan

            self.prob_v2g_everywhere.solve(solver='MOSEK', ignore_dpp=True, verbose=False)
            
            try:
                cost_v2g_everywhere[week_price] = self.obj_v2g_everywhere.value #- self.batt_deg_cost_v2g.value
            except:
                cost_v2g_everywhere[week_price] = np.nan

        #save the costs
        pd.DataFrame(cost_uncontrolled).to_csv(self.results_path + 'Results/'+self.region + '/batt_aging_'+str(self.batt_cost) + '/elrp_'+str(self.elrp_seed)+'_cost_uncontrolled_'+self.min_date_price + '_to_' + self.max_date_price+'_vin_'+str(vin)+'.csv')
        pd.DataFrame(cost_managed).to_csv(self.results_path + 'Results/'+self.region+ '/batt_aging_'+ str(self.batt_cost) + '/elrp_'+str(self.elrp_seed)+'_cost_managed_'+self.min_date_price + '_to_' + self.max_date_price +'_vin_'+str(vin)+'.csv')
        pd.DataFrame(cost_v2g_home).to_csv(self.results_path + 'Results/'+self.region+ '/batt_aging_'+str(self.batt_cost)  + '/elrp_'+str(self.elrp_seed)+'_cost_v2g_home_'+self.min_date_price + '_to_' + self.max_date_price +'_vin_'+str(vin)+'.csv')
        pd.DataFrame(cost_v2g_everywhere).to_csv(self.results_path + 'Results/'+self.region+ '/batt_aging_'+str(self.batt_cost)  + '/elrp_'+str(self.elrp_seed)+'_cost_v2g_everywhere_'+self.min_date_price + '_to_' + self.max_date_price +'_vin_'+str(vin)+'.csv')
        return
    
    def call_optimization_day_ahead(self, vin):
        """
        Runs V2G Home for all price weeks for the input vin with day ahead forecasting
        
        Args:
        vin: particular vin ID to run optimization for
        """ 

        #define cvxpy problems with parameters
        self.num_timesteps = int(24 / self.timestep_opt) #daily
        self.data_week_driving_soc_change = cp.Parameter(self.num_timesteps)
        self.combined_access = cp.Parameter(self.num_timesteps)
        self.price_week = cp.Parameter(self.num_timesteps)
        self.price_week_elrp = cp.Parameter(self.num_timesteps)
        self.public_chg_price_week = cp.Parameter(self.num_timesteps)
        self.tou_price_week = cp.Parameter(self.num_timesteps)
        self.uncontrolled_power_home = cp.Parameter(self.num_timesteps, nonneg=True)
        self.uncontrolled_power_away = cp.Parameter(self.num_timesteps, nonneg=True)
        self.max_power_away = cp.Parameter(self.num_timesteps, nonneg=True)
        self.max_power_home = cp.Parameter(self.num_timesteps, nonneg=True)
        self.access = cp.Parameter(self.num_timesteps, nonneg=True)
        self.home = cp.Parameter(self.num_timesteps, nonneg=True)
        self.soc_init_v2g_home = cp.Parameter(nonneg=True)
        self.soc_init_v2g_everywhere = cp.Parameter(nonneg=True)

        self.calendar_aging_coeffs()
        self.run_optimization_v2g_home(soc_specified=True)
        self.run_optimization_v2g_everywhere(soc_specified=True)
        
        data_uncontrolled = pd.read_csv(self.data_path +'/all_uncontrolled_demand_individualdrivers_'+self.min_date_vehicle + '_to_' + self.max_date_vehicle + '.csv')
        data_uncontrolled.datetime = pd.to_datetime(data_uncontrolled.datetime)
        mondays, num_weeks = self.define_weeks(self.min_date_vehicle, self.max_date_vehicle, data_uncontrolled)
        del data_uncontrolled
        sundays = mondays + datetime.timedelta(days=6)

        mondays_price, num_weeks = self.define_weeks(self.min_date_price, self.max_date_price, self.prices)
        cost_v2g_home = np.full((num_weeks*7,), np.nan)

        print('VIN:', vin)
        #find which weeks have data for this particular vin and create week num vector
        uncontrolled_data = pd.read_csv(self.data_path +'/all_uncontrolled_demand_individualdrivers_'+self.min_date_vehicle + '_to_' + self.max_date_vehicle +'.csv')
        uncontrolled_data.datetime = pd.to_datetime(uncontrolled_data.datetime)
        valid_vin_weeks = []
        for week_vehicle, monday in enumerate(mondays):
            monday_midnight = datetime.datetime.combine(monday, datetime.time.min)
            idx = np.where(uncontrolled_data.datetime == monday_midnight)[0][0]
            if np.sum(uncontrolled_data.iloc[idx:idx+24*7*60,int(vin)+1], axis=0)>20*60:
                valid_vin_weeks.append(week_vehicle)
        del uncontrolled_data

        #repeat vin-weeks until this vin has data for every week
        vin_weeks_vector = np.array(valid_vin_weeks)
        vin_weeks_vector = np.tile(vin_weeks_vector, 60)
        vin_weeks_vector = vin_weeks_vector[:num_weeks]

        driver = int(vin)
        
        #loop through each price_week
        for week_price, start in enumerate(mondays_price):
            week = vin_weeks_vector[week_price]
            print('Price Week:', week_price + 1, 'of', num_weeks)

            #get the dynamic price vector for the price week
            price_week = self.prices.loc[(self.prices.datetime.dt.date >= start)&(self.prices.datetime.dt.date <= start + datetime.timedelta(days=6))].Price.values

            #we want the prices for every minute, so need to upsample
            self.price_week_tmp = np.repeat(price_week / 100, 60)

            if self.elrp:
                #set up the ELRP price vector for the price week
                tmp_elrp = self.price_week_tmp
                #pick 10 random days to be ELRP days
                np.random.seed(self.elrp_seed)
                elrp_days = np.random.choice(np.arange(121,self.elrp_total_num_days+121), 10, replace=False)
                elrp_start_times = np.random.choice([16, 17, 18], 10, replace=True)
                #loop through each day to check if it's an ELRP day
                for i in range(7):
                    day = start + datetime.timedelta(days=i)
                    if day.timetuple().tm_yday in elrp_days:
                        start_time = i*24*60 + elrp_start_times[np.where(elrp_days==day.timetuple().tm_yday)[0][0]]*60
                        tmp_elrp[start_time:start_time + 3 * 60 ] = self.elrp_compensation
                self.price_week_elrp_tmp = tmp_elrp

            #get the fixed TOU price vector for the price week
            if self.tou == 'EV2A':
                tmp = np.zeros((24*7,))
                for i in range(7):
                    day = start + datetime.timedelta(days=i)
                    if day.month >= 6 and day.month <= 9:
                        tmp[i*24:(i+1)*24] = self.ev2a_summer.iloc[:,0].values
                    else:
                        tmp[i*24:(i+1)*24] = self.ev2a_winter.iloc[:,0].values
                self.tou_price_week_tmp = np.repeat(tmp, 60)

            #set the public charging price vector for the price week
            self.public_chg_price_week.value = self.public_chg_price * np.ones((self.num_timesteps,))

            data_uncontrolled = pd.read_csv(self.data_path + '/all_uncontrolled_demand_individualdrivers_'+self.min_date_vehicle + '_to_' + self.max_date_vehicle + '.csv')
            data_uncontrolled.datetime = pd.to_datetime(data_uncontrolled.datetime)
            datetime_vec =data_uncontrolled.datetime
            data_week_uncontrolled = data_uncontrolled.loc[(datetime_vec.dt.date >= mondays[week])&(datetime_vec.dt.date <= sundays[week]), str(driver)]
            del data_uncontrolled

            data_access = pd.read_csv(self.data_path +'/access_individualdrivers_'+self.min_date_vehicle + '_to_' + self.max_date_vehicle + '.csv')
            data_week_access = data_access.loc[(datetime_vec.dt.date >= mondays[week])&(datetime_vec.dt.date <= sundays[week]), str(driver)]
            del data_access

            data_driving_soc_change = pd.read_csv(self.data_path + '/driving_soc_change_'+self.min_date_vehicle + '_to_' + self.max_date_vehicle + '.csv')
            data_week_driving_soc_change = data_driving_soc_change.loc[(datetime_vec.dt.date >= mondays[week])&(datetime_vec.dt.date <= sundays[week]), str(driver)]
            del data_driving_soc_change

            data_binned_max_kW = pd.read_csv(self.data_path + '/binned_max_kW_'+self.min_date_vehicle + '_to_' + self.max_date_vehicle + '.csv')
            data_week_binned_max_kW = data_binned_max_kW.loc[(datetime_vec.dt.date >= mondays[week])&(datetime_vec.dt.date <= sundays[week]), str(driver)]
            del data_binned_max_kW

            data_home = pd.read_csv(self.data_path +'/home_'+self.min_date_vehicle + '_to_' + self.max_date_vehicle + '.csv')
            data_week_home = data_home.loc[(datetime_vec.dt.date >= mondays[week])&(datetime_vec.dt.date <= sundays[week]), str(driver)]
            del data_home

            self.uncontrolled_home = data_week_home
            self.uncontrolled_away  = data_week_access.astype(bool) & ~data_week_home.astype(bool)
            self.access_tmp = np.array(data_week_access.astype(bool))
            self.home_tmp = np.array(data_week_home.astype(bool))

            #set up necessary charging combinations:
            self.uncontrolled_power_away_tmp = np.array(data_week_uncontrolled * self.uncontrolled_away)
            self.uncontrolled_power_home_tmp = np.array(data_week_uncontrolled * self.uncontrolled_home)

            data_week_binned_max_kW[data_week_binned_max_kW.isna()] = 0

            self.data_week_driving_soc_change_tmp = np.array(data_week_driving_soc_change)
            self.max_power_away_tmp = np.array(data_week_binned_max_kW * (np.array(data_week_access) & ~data_week_home.astype(bool)))
            self.max_power_home_tmp = np.array(data_week_binned_max_kW * np.array(data_week_home))

            del data_week_binned_max_kW
            del data_week_access
            del data_week_home
            del data_week_uncontrolled
            del data_week_driving_soc_change

            self.soc_init_v2g_home.value = .50

            for day in range(7):
                self.price_week.value = self.price_week_tmp[day*24*60:(day+1)*24*60]
                self.max_power_away.value = self.max_power_away_tmp[day*24*60:(day+1)*24*60]
                self.max_power_home.value = self.max_power_home_tmp[day*24*60:(day+1)*24*60]
                self.uncontrolled_power_away.value = self.uncontrolled_power_away_tmp[day*24*60:(day+1)*24*60]
                self.uncontrolled_power_home.value = self.uncontrolled_power_home_tmp[day*24*60:(day+1)*24*60]
                self.data_week_driving_soc_change.value = self.data_week_driving_soc_change_tmp[day*24*60:(day+1)*24*60]
                self.access.value = self.access_tmp[day*24*60:(day+1)*24*60]
                self.home.value = self.home_tmp[day*24*60:(day+1)*24*60]
                self.price_week_elrp.value = self.price_week_elrp_tmp[day*24*60:(day+1)*24*60]
                self.tou_price_week.value = self.tou_price_week_tmp[day*24*60:(day+1)*24*60]
                
                self.prob_v2g.solve(solver='MOSEK', ignore_dpp=True, verbose=False)
                
                try:
                    cost_v2g_home[week_price*7 + day] = self.obj_v2g.value
                    self.soc_init_v2g_home.value = self.soc_v2g.value[-1]
                except:
                    cost_v2g_home[week_price*7 + day] = np.nan
                    self.soc_init_v2g_home.value = 0.5
                
        #save the costs
        pd.DataFrame(cost_v2g_home).to_csv(self.results_path + 'Results/'+self.region+ '/batt_aging_'+str(self.batt_cost)  + '/elrp_'+str(self.elrp_seed)+'_cost_v2g_home_'+self.min_date_price + '_to_' + self.max_date_price +'_vin_'+str(vin)+'day_ahead_' + str(random_noise) + 'random.csv')
        return

    def baseline_cost(self, data, price):
        '''Calculate cost of baseline charging given an input price'''
        return (data / self.eta) @ (price * self.timestep_opt)
    
    def batt_cost_vec(self, power, soc):
        '''Calculate battery degradation cost as a sum of cyclic and calendar aging'''
        power_aging_intersect = 150
        T =   2 * np.abs(power_aging_intersect) ** 0.5 + 15.1
        
        c_rate = np.abs(power_aging_intersect)  / (self.battcap  ) 
        current = np.abs(power_aging_intersect * 1000) / self.nom_voltage / self.parallel
        B1 = (self.a * np.power((T+273),2) + self.b * (T+273) + self.c)
        B2 = (self.d * (T+273) + self.e)
        aging_150 = B1 * cp.exp(B2 * c_rate) * current * self.timestep_opt
        cyclic_loss = (  aging_150 / (power_aging_intersect)**2) * cp.power(power, 2)
        calendar_loss =  self.a1 * cp.abs(power) + self.b1 * soc[:(self.num_timesteps)] + self.c1
        return  cyclic_loss, calendar_loss, (calendar_loss + cyclic_loss) *  (self.batt_cost * self.battcap) / self.eol
    

    def calendar_aging_coeffs(self):
        '''Calculate linearized calendar aging equation coefficients'''
        f840=self.alpha*np.exp(-self.Ea/(self.R*(40+273)))*.2272*(1/60/24)
        f950=(self.gamma*50+self.delta) * .2272*(1/60/24)
        c0=(f840+f950)/2
        coeff = c0/(f840*f950) 
        a = cp.Variable(1)
        b = cp.Variable(1)
        c = cp.Variable(1)
        pwr_opt = np.linspace(0,50,100)
        soc = np.linspace(0,100,100)
        obj = cp.sum_squares((a * pwr_opt + b * soc +c)-coeff*self.calendar_aging_fxn_1(pwr_opt)*self.calendar_aging_fxn_2(soc))
        prob = cp.Problem(cp.Minimize(obj), [])
        prob.solve()
        self.a1 = a.value
        self.b1 = b.value
        self.c1 = c.value
        return

    def calendar_aging_fxn_1(self,pwr):
        '''Returns calendar aging term as a function of power'''
        T = 2*np.abs(pwr)**.5+15.1
        return self.alpha*np.exp(-self.Ea/(self.R*(T+273)))*.2272*(1/60/24)

    def calendar_aging_fxn_2(self,soc):
        '''Returns calendar aging term as a function of SOC'''
        return (self.gamma*soc+self.delta) *.2272*(1/60/24)

    def run_optimization_managed(self):
        '''Run optimization for V1G managed charging'''
        
        charging_schedule_home = cp.Variable(self.num_timesteps,)
        charging_schedule_away = cp.Variable(self.num_timesteps,)
        self.charging_schedule_managed = charging_schedule_home + charging_schedule_away
        self.soc_managed = cp.Variable(self.num_timesteps+1,)
        charging_cost_away = (charging_schedule_away / self.eta) @ (self.public_chg_price_week * self.timestep_opt)
        charging_cost_home = (charging_schedule_home / self.eta) @ (self.tou_price_week * self.timestep_opt)
        self.charging_cost_managed = charging_cost_home + charging_cost_away

        self.power_managed = ((self.soc_managed[:-1] - self.soc_managed[1:])/ 100) * (self.battcap / self.timestep_opt)
        self.cyclic_managed, self.calendar_managed, self.batt_deg_cost_managed = self.batt_cost_vec(self.power_managed, self.soc_managed)

        #define objective function (in units of $)
        lam=0.0005
        self.obj_managed = self.charging_cost_managed +  lam * cp.sum_squares(self.charging_schedule_managed) + cp.sum(self.batt_deg_cost_managed) 

        #define constraints
        constraints = []
        constraints += [charging_schedule_home >= 0]
        constraints += [charging_schedule_away >= 0]
        constraints += [self.soc_managed[0]==self.soc_managed[-1], self.soc_managed >= 0, self.soc_managed<=100]
        constraints += [self.soc_managed[1:] == self.soc_managed[:-1] + 100*(1/self.battcap)*self.timestep_opt*self.charging_schedule_managed + self.data_week_driving_soc_change]
        constraints += [charging_schedule_away <= (self.max_power_away)/self.eta] #accounting for inefficiency 
        constraints += [charging_schedule_home <= (self.max_power_home)/self.eta] #accounting for inefficiency 
        constraints += [charging_schedule_home<= 12/self.eta]
  
        self.prob_managed = cp.Problem(cp.Minimize(self.obj_managed), constraints)
        
        return 
    

    def run_optimization_v2g_home(self, soc_specified = False):
        '''Run optimization with V2G allowed at home only'''
        
        #define cvxpy variables
        self.charging_schedule_home_v2g = cp.Variable(self.num_timesteps,)
        self.charging_schedule_away_v2g = cp.Variable(self.num_timesteps,)
        self.charging_schedule_v2g = cp.Variable(self.num_timesteps,)
        self.soc_v2g = cp.Variable(self.num_timesteps+1,)
        self.grid_discharge_v2g  = cp.Variable(self.num_timesteps,)

        #define power and degradation cost variables    
        self.power_v2g = (self.soc_v2g[:-1] - self.soc_v2g[1:])/ 100 *self.battcap / self.timestep_opt
        self.cyclic_v2g, self.calendar_v2g, self.batt_deg_cost_v2g = self.batt_cost_vec(self.power_v2g, self.soc_v2g)
        
        #set the charging cost depending on whether ELRP is enabled or not
        if self.elrp:
            charging_cost_home = (self.charging_schedule_home_v2g/self.eta - self.grid_discharge_v2g  * self.eta - self.uncontrolled_power_home / self.eta) @ (self.price_week_elrp * self.timestep_opt)
            charging_cost_home += (self.uncontrolled_power_home / self.eta) @ (self.tou_price_week * self.timestep_opt)
        else: 
            charging_cost_home = ((self.charging_schedule_home_v2g-self.grid_discharge_v2g - self.uncontrolled_power_home) / self.eta) @ (self.price_week * self.timestep_opt)
            charging_cost_home += (self.uncontrolled_power_home / self.eta) @ (self.tou_price_week * self.timestep_opt)
        charging_cost_away = (self.charging_schedule_away_v2g / self.eta) @ (self.public_chg_price_week * self.timestep_opt)

        #set the objective function (in units of $)
        self.obj_v2g = charging_cost_home + charging_cost_away + cp.sum(self.batt_deg_cost_v2g)
            
        #define constraints
        constraints = []

        #charging schedule constraints    
        constraints += [self.charging_schedule_home_v2g >= 0]
        constraints += [self.charging_schedule_v2g == self.charging_schedule_home_v2g + self.charging_schedule_away_v2g]
        constraints += [self.charging_schedule_away_v2g >= 0]
        constraints += [self.charging_schedule_v2g >= 0]
        constraints += [self.charging_schedule_home_v2g <= self.home * self.v2g_max / self.eta]  
        constraints += [self.charging_schedule_away_v2g <= (self.max_power_away)/self.eta] #accounting for inefficiency 

        #SOC constraints
        if soc_specified:
            constraints += [self.soc_v2g[0] == self.soc_init_v2g_home, self.soc_v2g[-1] == self.soc_init_v2g_home]
        else:
            constraints += [self.soc_v2g[0]==self.soc_v2g[-1]]
        constraints += [self.soc_v2g >= 0, self.soc_v2g <= 100]
        constraints += [self.soc_v2g[1:] == self.soc_v2g[:-1] + 100*(1/self.battcap)*self.timestep_opt*self.charging_schedule_v2g  - 100*(1/self.battcap)*self.timestep_opt*self.grid_discharge_v2g+ self.data_week_driving_soc_change]
                
        #energy export constraints
        constraints += [self.grid_discharge_v2g >= 0]
        constraints += [self.grid_discharge_v2g <= 12]
        constraints += [self.grid_discharge_v2g <= self.v2g_max *self.home * self.eta]    

        #set up the problem
        self.prob_v2g = cp.Problem(cp.Minimize(self.obj_v2g), constraints)
        
        return
    


    def run_optimization_v2g_everywhere(self, soc_specified = False):
        '''Run optimization for V2G allowed everywhere'''
        
        #define cvxpy variables
        self.charging_schedule_home_v2g_everywhere = cp.Variable(self.num_timesteps,)
        self.charging_schedule_away_v2g_everywhere = cp.Variable(self.num_timesteps,)
        self.charging_schedule_v2g_everywhere = cp.Variable(self.num_timesteps,)
        self.soc_v2g_everywhere = cp.Variable(self.num_timesteps+1,)
        self.grid_discharge_v2g_everywhere  = cp.Variable(self.num_timesteps,)

        #define power and degradation cost variables
        self.power_v2g_everywhere = (self.soc_v2g_everywhere[:-1] - self.soc_v2g_everywhere[1:])/ 100 *self.battcap / self.timestep_opt
        self.cyclic_v2g_everywhere, self.calendar_v2g_everywhere, self.batt_deg_cost_v2g_everywhere = self.batt_cost_vec(self.power_v2g_everywhere, self.soc_v2g_everywhere)
        
        #set the charging cost depending on whether ELRP is enabled or not
        if self.elrp:
            charging_cost = (self.charging_schedule_v2g_everywhere/self.eta - self.grid_discharge_v2g_everywhere  * self.eta - (self.uncontrolled_power_home + self.uncontrolled_power_away) / self.eta) @ (self.price_week_elrp * self.timestep_opt)
            charging_cost += ((self.uncontrolled_power_home + self.uncontrolled_power_away)/ self.eta) @ (self.tou_price_week * self.timestep_opt)
        else: 
            charging_cost = (self.charging_schedule_v2g_everywhere/self.eta - self.grid_discharge_v2g_everywhere  * self.eta - (self.uncontrolled_power_home + self.uncontrolled_power_away) / self.eta) @ (self.price_week * self.timestep_opt)
            charging_cost += ((self.uncontrolled_power_home + self.uncontrolled_power_away)/ self.eta) @ (self.tou_price_week * self.timestep_opt)

        #define objective function (in units of $)
        self.obj_v2g_everywhere = charging_cost + cp.sum(self.batt_deg_cost_v2g_everywhere)
            
        #define constraints
        constraints = []

        #charging schedule constraints
        constraints += [self.charging_schedule_home_v2g_everywhere >= 0]
        constraints += [self.charging_schedule_v2g_everywhere == self.charging_schedule_home_v2g_everywhere + self.charging_schedule_away_v2g_everywhere]
        constraints += [self.charging_schedule_away_v2g_everywhere >= 0]
        constraints += [self.charging_schedule_home_v2g_everywhere <= self.home * self.v2g_max / self.eta]  #restrict home charging to v2g max power
        constraints += [self.charging_schedule_away_v2g_everywhere <= (self.max_power_away)/self.eta] #accounting for inefficiency 
        constraints += [self.charging_schedule_v2g_everywhere >= 0]

        #SOC constraints
        if soc_specified:
            constraints += [self.soc_v2g_everywhere[0] == self.soc_init_v2g_everywhere, self.soc_v2g_everywhere[-1] == self.soc_init_v2g_everywhere]
        else:
            constraints += [self.soc_v2g_everywhere[0]==self.soc_v2g_everywhere[-1]]
        constraints += [self.soc_v2g_everywhere >= 0, self.soc_v2g_everywhere <= 100]
        constraints += [self.soc_v2g_everywhere[1:] == self.soc_v2g_everywhere[:-1] + 100*(1/self.battcap)*self.timestep_opt*self.charging_schedule_v2g_everywhere  - 100*(1/self.battcap)*self.timestep_opt*self.grid_discharge_v2g_everywhere+ self.data_week_driving_soc_change]

        #Energy export constraints
        constraints += [self.grid_discharge_v2g_everywhere <= self.v2g_max * self.access *self.eta]    #can discharge at max whenever connected to charger
        constraints += [self.grid_discharge_v2g_everywhere >= 0]
        
        #set up the problem
        self.prob_v2g_everywhere = cp.Problem(cp.Minimize(self.obj_v2g_everywhere), constraints)
        
        return


