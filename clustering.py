import os
import numpy as np
import pandas as pd
import sklearn as sk
from tslearn.clustering import TimeSeriesKMeans
import datetime
from charging import ChargingData
from charging import ChargingAutomation
import time

def load_data(filename):
    """
    Reads in main EV dataset. 

    Args:
        filename: string of dataset file location

    Returns:
        vehicle_df_all: A 2D DataFrame with vehicle data
    """
    vehicle_df_all = pd.read_csv(filename, index_col=0, na_values=" ", engine='python')
    print(vehicle_df_all.head())
    return vehicle_df_all

def create_baseline_data(vehicle_df_all, min_date, max_date):
    """
    Sets simulation date range and get uncontrolled demand data. This function should only be run once.

    Args:
        vehicle_df_all: A 2D DataFrame with vehicle data
        min_date: datetime date of start date for vehicle data
        max_date: datetime date of end date for vehicle data
    """

    period_string = str(min_date) + '_to_' + str(max_date)

    #set simulation weeks
    data = ChargingData(vehicle_df_all.copy(deep=True))
    data.define_weeks(min_date=min_date, max_date=max_date)

    save_str = 'Data'+'/'+'all_uncontrolled_demand'

    #calculate baseline demand for all the cars only once
    print('-----'*5)
    print('Baseline Demand Computation')        

    automation = ChargingAutomation(min_date, max_date, data=data)

    for week in range(data.num_weeks):
        tic = time.time()
        print('Week starting on : ', data.mondays[week])
        automation.calculate_uncontrolled_only_oneweek(week, verbose=True)
        toc = time.time()
        print('Elapsed time: ', toc-tic)

    cols = automation.uncontrolled_charging_demand.columns[1:]

    #save access, baseline demand, binned max kW, driving soc change, and soc value files
    automation.access.rename(columns = {i:str(int(i)) for i in cols}, inplace = True)
    automation.access.to_csv('Data/'+'access_individualdrivers_'+period_string+'.csv')

    automation.uncontrolled_charging_demand.rename(columns = {i:str(int(i)) for i in cols}, inplace = True)
    automation.uncontrolled_charging_demand.to_csv(save_str+'_individualdrivers_'+period_string+'.csv')
    
    automation.binned_max_kW.rename(columns = {i:str(int(i)) for i in cols}, inplace = True)
    automation.binned_max_kW.to_csv('Data/'+'binned_max_kW_'+period_string+'.csv')

    automation.driving_soc_change.rename(columns = {i:str(int(i)) for i in cols}, inplace = True)
    automation.driving_soc_change.to_csv('Data/'+'driving_soc_change_'+period_string+'.csv')

    automation.home.rename(columns = {i:str(int(i)) for i in cols}, inplace = True)
    automation.home.to_csv('Data/'+'home_'+period_string+'.csv')

    automation.uncontrolled_soc.rename(columns = {i:str(int(i)) for i in cols}, inplace = True)
    automation.uncontrolled_soc.to_csv('Data/'+'uncontrolled_soc_'+period_string+'.csv')

    return

def define_weeks(data, min_date, max_date):
    """
    Finds all Mondays in input date range.

    Args:
        data: A 2D DataFrame with vehicle data
        min_date: A datetime object indicating the start date of the simulation
        max_date: A datetime object indicating the end date of the simulation

    Returns:
        mondays: An ndarray of datetime objects representing all Mondays in the input date range
    """
    
    mondays = np.sort(data.loc[(data.datetime.dt.weekday==0)&(data.datetime.dt.date >= min_date)&(data.datetime.dt.date < max_date)].datetime.dt.date.unique())
    return mondays


def create_feature_array(min_date, max_date, feature_type, path):
    """
    Creates feature array for clustering.

    Args:
        min_date: A datetime object indicating the start date of the simulation
        max_date: A datetime object indicating the end date of the simulation
        feature_type: String indicating type of feature

    Returns:
        feat_array: A 3D ndarray with dimensions (num driver-days, num_timesteps, num_features)
        vinids: A list of arrays with the vinids of the drivers included in the feature array
    """

    #read charging and access data
    period_string = str(min_date) + '_to_' + str(max_date)
    uncontrolled_data = pd.read_csv(path + 'all_uncontrolled_demand_individualdrivers_'+period_string+'.csv')
    access_data = pd.read_csv(path + 'access_individualdrivers_'+period_string+'.csv')
    
    uncontrolled_data.datetime = pd.to_datetime(uncontrolled_data.datetime)
    access_data.datetime = pd.to_datetime(access_data.datetime)

    mondays = define_weeks(uncontrolled_data, min_date, max_date)

    #only include drivers with more than 20kWh of charging demand for the indicated week
    #vinids will be a list of arrays
    vinids =[]

    if feature_type == 'access_daily_weekdays':
        feat_array = np.array(())

        #loop through each vinid in the dataset
        for vin in access_data.keys().values[2:]:
            access_array = np.array(())
            #loop through each week
            for monday in mondays:
                monday_midnight = datetime.datetime.combine(monday, datetime.time.min)
                idx = np.where(uncontrolled_data.datetime == monday_midnight)[0][0]
                #check if there is at least 20 kwh of charging demand during that week:
                if np.sum(uncontrolled_data.iloc[idx:idx+24*7*60,int(vin)+1], axis=0)>20*60:
                    vinids.append(int(vin))
                    #stack the days to make the access array
                    for day in range(5):
                        idx = idx+ 24*60
                        if access_array.shape[0] == 0:
                            access_array = access_data.iloc[idx:idx+24*60,int(vin)+1].values
                        else:
                            access_array = np.vstack((access_array, access_data.iloc[idx:idx+24*60,int(vin)+1].values))

            #take the mean of the access array 
            if feat_array.shape[0] == 0:
                feat_array = np.mean(access_array, axis=0)
            else:
                if access_array.shape[0] > 0:
                    feat_array = np.vstack((feat_array, np.mean(access_array, axis=0)))
                else:
                    pass

        vinids = np.sort(np.unique(np.array(vinids)))
        return feat_array, vinids

    else:
        print('Feature type not recognized. Ending function.')
        return None, None    


def run_adaptive_clustering(feat_array):
    """
    Clusters data using the adaptive k-means algorithm and saves labels and cluster centers.

    Args:
        feat_array: A 3D ndarray with dimensions (num drivers, num_timesteps, num_features)
    """
    #scale the feature vector using robust scaler, which is less sensitive to outliers
    scaler = sk.preprocessing.RobustScaler()
    scaler.fit(feat_array)
    feat_array_scaled = feat_array.copy()
    feat_array_scaled = scaler.transform(feat_array)

    #set clustering parameters
    min_k = 4
    max_k = 400
    theta=.1

    #initialize clustering algorithm
    K = min_k
    Nv = np.zeros(K)

    #run kmeans clustering
    kmeans = TimeSeriesKMeans(n_clusters=K, verbose=1).fit(feat_array_scaled)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    centers= scaler.inverse_transform(kmeans.cluster_centers_[:,:,0])
    iter=0

    #iterate and split clusters until there are no violations or until max number of clusters is reached
    while True:
        print('iteration', iter)
        iter+=1
        print('Number of clusters: ', K)

        #for each cluster with a violation, split into two clusters
        for k in np.where(Nv>0)[0]:
            feat_array_k = feat_array[labels==k,:]
            feat_array_k_scaled = feat_array_scaled[labels==k,:]
            kmeans_k = TimeSeriesKMeans(n_clusters=2, verbose=1).fit(feat_array_k_scaled)
            labels[labels==k] = kmeans_k.labels_ * K + (1-kmeans_k.labels_) * k
            centers_k = scaler.inverse_transform(kmeans_k.cluster_centers_[:,:,0])

            #update centers by inserting the first center of the new clusters to k and the next one K+1
            centers[k,:] = centers_k[0,:]
            centers = np.vstack((centers, centers_k[1,:].T))
            K += 1

        #loop through all clusters to compute the distance metric and track whether there are violations
        Nv = np.zeros(K)
        for k in range(K):
            feat_array_k = feat_array[labels==k,:]
            for row in range(feat_array_k.shape[0]):
                #compute the distance between the row and the cluster center
                dist = np.sum(np.power(feat_array_k[row,:] - centers[k,:], 2))
                #if the distance is greater than theta*c, then we have a violation
                condition = dist > theta * np.sum(np.power(centers[k,:], 2))
                if condition and feat_array_k.shape[0] > 2: #if there is more than one row in the cluster, we can count the violation
                    Nv[k] += 1

        
        #if violations == 0, then we can stop
        if np.sum(Nv) == 0:
            print('No violations, stopping clustering')
            break
        else: #run k means again for those clusters that have violations
            print('Violations found, running k-means again')
            

        if K >= max_k:
            print('Reached maximum number of clusters, stopping clustering')
            break
    print('Number of clusters after first algorithm:', K)
    print('Number of vehicles in each cluster:', np.bincount(labels))
  
    #combine clusters to ensure there are at least 10 vehicles in each cluster
    while True:
        print('Number of clusters: ', K)
        #find the smallest cluster
        a=np.bincount(labels)
        min_size = np.min(a[a>0])
        if min_size < 10: #if the smallest cluster is less than 10, we combine it with the closest cluster
            print('Smallest cluster size:', min_size)
            #find the smallest cluster that is not empty
            min_cluster = np.argmin(a[a>0])
            unique_clusters = np.unique(labels)
            
            #find the closest cluster to the smallest cluster
            distances = np.zeros(K)
            for k in range(K):
                if k != min_cluster:
                    distances[k] = np.sum(np.power(centers[min_cluster,:] - centers[k,:], 2))
            closest_cluster = np.argmin(distances[distances>0])
            
            #merge the two clusters
            ni = np.sum(labels==unique_clusters[min_cluster])
            nj = np.sum(labels==unique_clusters[closest_cluster])
            print('Number of points in cluster', min_cluster, ':', ni)
            print('Number of points in cluster', closest_cluster, ':', nj)

            #weighted average the centers
            centers[min_cluster,:] = (ni*centers[min_cluster,:] + nj*centers[closest_cluster,:]) / (ni+nj)

            #remove the closest cluster center
            centers = np.delete(centers, closest_cluster, axis=0)

            #reset the labels for the closest cluster
            labels[labels==unique_clusters[closest_cluster]] = unique_clusters[min_cluster]   
            K -= 1
        else:
            print('All clusters are larger than 10, stopping clustering')
            break
                    
    #save the labels and centers
    os.makedirs('Data/Cluster_Results_Adaptive', exist_ok=True)
    np.savetxt('Data/Cluster_Results_Adaptive/cluster_labels_'+str(K)+'.csv', labels, delimiter=",")
    np.savetxt('Data/Cluster_Results_Adaptive/cluster_centers_'+str(K)+'.csv', centers, delimiter=",")

    return 


