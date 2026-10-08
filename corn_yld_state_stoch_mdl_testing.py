#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 09:59:53 2022

TO DO:
    1) Evaluate the impact/benefit of getting the data pulls outside of the loop.
    

@author: michaelmcconnell
"""

#import all needed libraries and modules
import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

print('Modules imported')

# Import hostircal WASDE data from 2010 to 2020

wasde2010 ='https://www.usda.gov/sites/default/files/documents/oce-wasde-report-data-2010-04-to-2015-12.zip'
wasde2016 = 'https://www.usda.gov/sites/default/files/documents/oce-wasde-report-data-2016-01-to-2020-12.zip'
wasdeJun21 = 'https://www.usda.gov/sites/default/files/documents/oce-wasde-report-data-2021-06.csv'
wasdeJun22 = 'https://www.usda.gov/sites/default/files/documents/oce-wasde-report-data-2022-06.csv'
wasdeApr21 = 'https://www.usda.gov/sites/default/files/documents/oce-wasde-report-data-2021-04.csv'
wasdeApr22 = 'https://www.usda.gov/sites/default/files/documents/oce-wasde-report-data-2022-04.csv'
wasdeApr23 = 'https://www.usda.gov/sites/default/files/documents/oce-wasde-report-data-2023-04.csv'
wasdeNov23 = 'https://www.usda.gov/sites/default/files/documents/oce-wasde-report-data-2023-11.csv'



wasde_list= [wasde2010, wasde2016, wasdeJun21, wasdeJun22, wasdeApr21, wasdeApr22, wasdeApr23, wasdeNov23]
arch=pd.DataFrame()


# arch1 = pd.read_csv(wasde2010)
# arch2 = pd.read_csv(wasde2016)
for w in wasde_list:
    arch1 = pd.read_csv(w)
    arch = pd.concat([arch, arch1])
    
qry='Commodity=="Corn" & Attribute=="Yield per Harvested Acre" & ForecastMonth.isin([6,4]) & ProjEstFlag=="Proj."'
hist_ylds = arch.query(qry)[["MarketYear", "Value", "ForecastMonth"]].pivot_table(values='Value', index='MarketYear', columns='ForecastMonth').reset_index().rename(columns={4:'Final', 6:'June'})
hist_ylds['mdl_year']=hist_ylds['MarketYear'].str.slice(0,4).astype('int64')

fcast_year_list = range(2000, 2024, 1)
stoch_mdl_stats_dict={}

# import the relevent data

for fc_yr in fcast_year_list:

    # Defined variables and objects
    mdl_start_year = 1980
    mdl_forecast_year = fc_yr ## THIS IS WHAT NEEDS TO BE LOOPED THROUGH IN THE TESTING PROCESS
    plt_prog_toggle = True # toggles between using actual planting progress (True) or force "normal" adjusted planting progress rate of 1 (False)
    plt_prog_month = 5
    plt_prog_day = 20 #Current limited to May 21 due to limitations of reported planting progress by state in some years after this date, which results in dropping years from datasets
    num_of_draws = 10000 #use >10,000 draws for production runs
    weather_draw_start_year = 1895
    
    print('variables set')
    
    #import weather data
    
    ## Date associated with files
    procdate = 'https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/procdate.txt'
    date = requests.get(procdate).text[:]
    
    ## data url structures (to be added with the date to get the complete URL)
    
    pcpn_st_url = 'https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-pcpnst-v1.0.0-'
    tmax_st_url = 'https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-tmaxst-v1.0.0-'
    tmin_st_url = 'https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-tminst-v1.0.0-'
    tmpc_st_url = 'https://www1.ncdc.noaa.gov/pub/data/cirs/climdiv/climdiv-tmpcst-v1.0.0-'
    
    st_list = []
    st_url_list = [pcpn_st_url, tmax_st_url, tmin_st_url, tmpc_st_url]
    
    for i in st_url_list:
        fmt_url = str(i) + str(date)
        st_list.append(fmt_url)
    
    # Import and format data
    
    cols = ['code', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    element_dict = {
        '01' : 'Precipitation', '02' : 'Average Temperature', '05' : 'PDSI', '06' : 'PHDI', '07' : 'ZNDX',
        '08' : 'PMDI', '25' : 'Heating Degree Days', '26' : 'Cooling Degree Days', '27' : 'Maximum Temperature',
        '28' : 'Minimum Temperature', '71' : '1-month Standardized Precipitation Index', '72' : '2-month Standardized Precipitation Index',
        '73' : '3-month Standardized Precipitation Index', '74' : '6-month Standardized Precipitation Index',
        '75' : '9-month Standardized Precipitation Index', '76' : '12-month Standardized Precipitation Index',
        '77' : '24-month Standardized Precipitation Index'
        }
    
    state_dict = {
        '001':'Alabama', '030': 'New York',
        '002':'Arizona', '031':'North Carolina',
        '003':'Arkansas', '032':'North Dakota',
        '004':'California', '033':'Ohio',
        '005':'Colorado', '034':'Oklahoma',
        '006':'Connecticut', '035':'Oregon',
        '007':'Delaware', '036':'Pennsylvania',
        '008':'Florida', '037':'Rhode Island',
        '009':'Georgia','038':'South Carolina',
        '010':'Idaho', '039':'South Dakota',
        '011':'Illinois', '040':'Tennessee',
        '012':'Indiana', '041':'Texas',
        '013':'Iowa', '042':'Utah',
        '014':'Kansas', '043':'Vermont',
        '015':'Kentucky', '044':'Virginia',
        '016':'Louisiana', '045':'Washington',
        '017':'Maine', '046':'West Virginia',
        '018':'Maryland', '047':'Wisconsin',
        '019':'Massachusetts', '048':'Wyoming',
        '020':'Michigan','050':'Alaska',     
        '021':'Minnesota', '101':'Northeast Region',
        '022':'Mississippi', '102':'East North Central Region',
        '023':'Missouri', '103':'Central Region',
        '024':'Montana', '104':'Southeast Region',  
        '025':'Nebraska', '105':'West North Central Region', 
        '026':'Nevada', '106':'South Region',
        '027':'New Hampshire', '107':'Southwest Region',  
        '028':'New Jersey', '108':'Northwest Region',  
        '029':'New Mexico', '109':'West Region',
        '110':'National (contiguous 48 States)',
        
        '250':'Spring Wheat Belt (area weighted)',
        '255':'Primary Hard Red Winter Wheat Belt (area weighted)',
        '256':'Winter Wheat Belt (area weighted)',
        '260':'Primary Corn and Soybean Belt (area weighted)',
        '261':'Corn Belt (area weighted)',
        '262':'Soybean Belt (area weighted)',
        '265':'Cotton Belt (area weighted)',
        
        }
    
    st_mly = pd.DataFrame(columns=['date', 'code', 'state', 'element', 'div', 'year', 'month', 'value'])
    
    nan_list = ['-99.99', '-99.90', '99.90', '-9.99', '-9999', '-99.99', '9.99']
    
    col_specs = [(0, 10), (11, 17), (18, 24), (25, 31), (32, 38), (39, 45), (46, 52), (53, 59), (60, 66), (67, 73), (74, 80), (81, 87), (88, 94)]
    
    for i in st_list:
        noaa_df = pd.read_fwf(i, colspecs=col_specs, engine='python', header=None, dtype='str')
        noaa_df = noaa_df.replace(nan_list, np.nan)
        noaa_df.columns = cols
        noaa_df = noaa_df.melt(id_vars = ['code'], var_name='month', value_name='value')
        noaa_df['state'] = noaa_df['code'].str.slice(0, 3)
        noaa_df['div'] = noaa_df['code'].str.slice(3, 4)
        noaa_df['element'] = noaa_df['code'].str.slice(4, 6)
        noaa_df['year'] = noaa_df['code'].str.slice(6, )
        noaa_df['date'] = pd.to_datetime(noaa_df['year'] + noaa_df['month'] + '01')
        noaa_df['value'] = noaa_df['value'].astype('float64')
        noaa_df['id'] = noaa_df['code'] + noaa_df['month']
        noaa_df = noaa_df.sort_values(['year', 'month', 'state'])
        st_mly = pd.concat([st_mly, noaa_df])
        print('Success: ' + str(i))
    
    st_mly = st_mly.set_index('date')
    st_mly['element_desc'] = st_mly['element'].map(element_dict)
    st_mly['state_name'] = st_mly['state'].map(state_dict)
    
    print(st_mly.info())
        
    #api_key_nass = os.environ.get('NASS_API_KEY')
    api_key_nass = '90AAEBEA-E090-3723-A14B-D53458AD46D1'
    
    nass_get_base = 'http://quickstats.nass.usda.gov/api/api_GET/'
    
    def PL_QS_pull(Payload):
        '''
        Takes a payload for a NASS QuickStats
        API and returns a pandas DataFrame with
        Values column formatted as a numeric object
        '''
        r = requests.get(nass_get_base, params=Payload)
        r_json = r.json()   
        r_df = pd.DataFrame.from_dict(r_json['data']) 
        #Reformat DF so that data types are appropriate
        r_df['Value'] = r_df['Value'].str.replace(",", "") #need this otherwise to_numeric function converts all to NaN
        r_df['Value'] = pd.to_numeric(r_df['Value'], errors='coerce')
        
        return r_df
    print('Complete')
    
    ## Payload and API call for annual data
    
    prod_PL = {
        'key': api_key_nass,
        'source_desc':'SURVEY',
        'agg_level_desc':['STATE', 'NATIONAL'],
        'statisticcat_desc':['YIELD', 'AREA PLANTED', 'AREA HARVESTED'],
        'commodity_desc':'CORN',
        'reference_period_desc':['YEAR', 'YEAR - MAR ACREAGE'],  ## NEED TO ADD “YEAR - MAR ACREAGE” PARAMETER FOR TESTING, BUT DATA ONLY AVAILABLE IN QS STARTING IN 2012
        'prodn_practice_desc':'ALL PRODUCTION PRACTICES',
        'util_practice_desc':['ALL UTILIZATION PRACTICES', 'GRAIN'],
        'unit_desc':['BU / ACRE', 'ACRES'],
        'year__GE':'1895', 
        'format':'json'
        }
    prod_df = PL_QS_pull(prod_PL)
    prod_fmt = prod_df[
        (prod_df['year'] >= mdl_start_year) &
        (prod_df['reference_period_desc'].isin(['YEAR']))
        ]
    
    prod_mdl = prod_fmt.pivot_table(index=['year', 'state_name'], columns='statisticcat_desc', values='Value').reset_index()
    prod_mdl['state_name'] = prod_mdl['state_name'].str.title()
    prod_mdl = prod_mdl.rename(columns={'AREA HARVESTED':'ah', 'AREA PLANTED':'ap', 'YIELD':'yld', 'Value':'value'})
    
    print('Production data complete')
    
    prog_PL = {
        'key': api_key_nass,
        'source_desc':'SURVEY',
        'statisticcat_desc':['PROGRESS'],
        'commodity_desc':'CORN',
        'unit_desc':['PCT PLANTED'],
        'format':'json'
        }
    prog_df = PL_QS_pull(prog_PL)
    print('progress data completed')
    
    prog_fmt = prog_df[
        (prog_df['year']>=mdl_start_year) &
        (prog_df['year']<=mdl_forecast_year)
        ]
    prog_fmt.loc[:,'week_ending'] = pd.to_datetime(prog_fmt['week_ending'])
    prog_fmt.loc[:,'state_name'] = prog_fmt['state_name'].str.title()
    prog_fmt = prog_fmt.set_index('week_ending')
    prog_mdl = prog_fmt.groupby(['state_name', 'year'])['Value'].resample('D').mean().interpolate().reset_index() #augment to interpolate to 100 percent when reporting stops
    prog_mdl = prog_mdl.rename(columns={'Value':'plt_prog'})
    
    print('complete')
    
    median_prog = prog_mdl.assign(day = prog_mdl.week_ending.dt.day, month = prog_mdl.week_ending.dt.month)
    median_prog = median_prog.groupby(['state_name', 'month', 'day']).agg({'plt_prog':np.median}) # used for prediction matrices for all forecast estimates
    median_merge = median_prog.reset_index().rename(columns={'plt_prog':'plt_median'})
    
    prog_mdl = prog_mdl.assign(day = prog_mdl.week_ending.dt.day, month = prog_mdl.week_ending.dt.month)
    prog_mdl = prog_mdl.merge(median_merge, on=['state_name', 'month', 'day'], how='left')  #.drop(['month', 'day'], axis=1)
    prog_mdl['plt_adj'] = prog_mdl['plt_prog']/prog_mdl['plt_median']
    prog_mdl_filter = prog_mdl[(prog_mdl['month']==plt_prog_month) & (prog_mdl['day']==plt_prog_day)]
    print('complete')
    
    wx_df = st_mly.reset_index()
    print('weather file created')
    
    # format weather data
    
    wx_df['date'] = pd.to_datetime(wx_df['date'])
    
    wx_fmt = wx_df[
        (wx_df['date'].dt.month.isin([4, 5, 6, 7, 8])) 
        ]
    wx_fmt.loc[:,'col_labs'] = wx_fmt.loc[:,'element_desc'].map({
        'Average Temperature':'temp', 'Maximum Temperature':'max_temp', 'Minimum Temperature':'min_temp', 'Precipitation':'pcpn'
        })
    wx_fmt = wx_fmt.assign(
        year = wx_fmt['date'].dt.year,
        month = wx_fmt['date'].dt.month.astype(str)
        )
    wx_fmt = wx_fmt.pivot_table(index = ['year', 'month', 'state_name'], columns=['col_labs'], values='value').reset_index()
    wx_mdl = wx_fmt.pivot_table(index = ['year', 'state_name'], columns=['month'], values=['max_temp', 'min_temp', 'pcpn', 'temp']).reset_index()
    
    wx_mdl.columns = ['_'.join(col).strip() for col in wx_mdl.columns.values]
    
    wx_mdl = wx_mdl.assign(
        pcpn_6_sq = wx_mdl['pcpn_6']**2,
        pcpn_7_sq = wx_mdl['pcpn_7']**2,
        max_temp_7_sq = wx_mdl['max_temp_7']**2,
        pcpn_8_sq = wx_mdl['pcpn_8']**2,
        max_temp_8_sq = wx_mdl['max_temp_8']**2
        )
    wx_mdl = wx_mdl.rename(columns={'year_':'year', 'state_name_':'state_name'})
    print('weather formatting complete')
    
    # Merge dataframes for model
    
    mdl_df = prod_mdl.merge(prog_mdl_filter, how='left', on = ['year', 'state_name'])
    mdl_df = mdl_df.merge(wx_mdl, how='left', on=['year', 'state_name'])
    print('Weather and crop data merged')
    
    # Run state-level regression models
    
    state_list = ['Illinois', 'Iowa', 'Nebraska', 'Minnesota', 'Wisconsin', 
                  'South Dakota', 'Kansas', 'Missouri', 'Ohio', 
                  'Michigan', 'Indiana', 'Kentucky', 'Colorado']
    
    state_mdl_dict = {}
    state_yhat_df = pd.DataFrame()
    
    for s in state_list:    
        mdl_df0 = mdl_df[
            (mdl_df['state_name']==s) &
            (mdl_df['year']<mdl_forecast_year)
            ]    
        mdl = smf.ols('yld ~ year + plt_adj + pcpn_6 + pcpn_7 + pcpn_7_sq + max_temp_7 + pcpn_8 + max_temp_8', data=mdl_df0).fit()
        
        if plt_prog_toggle == False:
            plt_pred = 1
        elif prog_mdl['week_ending'].max() >= pd.to_datetime(str(mdl_forecast_year) + '-' + str(plt_prog_month) + '-' + str(plt_prog_day)):
            plt_pred = prog_mdl[(prog_mdl.year == mdl_forecast_year)&(prog_mdl['week_ending'].dt.month==plt_prog_month)&(prog_mdl['week_ending'].dt.day==plt_prog_day)&(prog_mdl['state_name']==str(s))]['plt_adj'].squeeze().astype('float64')
        else:
            plt_pred = 1
            
        pred_dict = {'year':[mdl_forecast_year], 
                     'plt_prog':[np.median(mdl_df0['plt_prog'])],
                     'plt_adj':[plt_pred],
                     'pcpn_6':[np.mean(mdl_df0['pcpn_6'])],
                     'pcpn_7':[np.mean(mdl_df0['pcpn_7'])],
                     'pcpn_7_sq':[np.mean(mdl_df0['pcpn_7'])**2],
                     'max_temp_7':[np.mean(mdl_df0['max_temp_7'])],
                     'pcpn_8':[np.mean(mdl_df0['pcpn_8'])],
                     'max_temp_8':[np.mean(mdl_df0['max_temp_8'])]}
        
        pred_df = pd.DataFrame(pred_dict)
        yld_hat_df = pred_df.assign(yld_hat = mdl.predict(pred_df), state_name=str(s))
        
        state_mdl_dict[str(s)] = [mdl.summary(), yld_hat_df, mdl]
        state_yhat_df = pd.concat([state_yhat_df, yld_hat_df[['state_name', 'yld_hat']]])
        
    
    print('Prog state regression estimates complete')
    
    # Model for states that don't have full planting progress data
    
    all_state_list = prod_df[prod_df['statisticcat_desc']=='AREA PLANTED']['state_name'].str.title().unique().tolist()
    all_state_list.remove('Us Total')
    nonprog_state_list = [x for x in all_state_list if x not in state_list]
    
    for s in nonprog_state_list:    
        try:
            mdl_df0 = mdl_df[
                (mdl_df['state_name']==s) &
                (mdl_df['year']<=mdl_forecast_year)
                ]    
            mdl = smf.ols('yld ~ year + pcpn_6 + pcpn_7 + pcpn_7_sq + max_temp_7 + pcpn_8 + max_temp_8', data=mdl_df0).fit()
            
            pred_dict = {'year':[mdl_forecast_year], 
                         'pcpn_6':[np.mean(mdl_df0['pcpn_6'])],
                         'pcpn_7':[np.mean(mdl_df0['pcpn_7'])],
                         'pcpn_7_sq':[np.mean(mdl_df0['pcpn_7'])**2],
                         'max_temp_7':[np.mean(mdl_df0['max_temp_7'])],
                         'pcpn_8':[np.mean(mdl_df0['pcpn_8'])],
                         'max_temp_8':[np.mean(mdl_df0['max_temp_8'])]}
            
            pred_df = pd.DataFrame(pred_dict)
            yld_hat_df = pred_df.assign(yld_hat = mdl.predict(pred_df), state_name=str(s))
            
            state_mdl_dict[str(s)] = [mdl.summary(), yld_hat_df, mdl]
            state_yhat_df = pd.concat([state_yhat_df, yld_hat_df[['state_name', 'yld_hat']]])
        except:
            pass
    
    print('Non-prog state regression estimates complete')
    
    # Develop planted-area state weights
    
    state_list_upper = [x.upper() for x in state_list]
    
    ## Adjustment for silage and abandonment
    grain_coef = prod_mdl[['state_name', 'year', 'ap', 'ah']][prod_mdl['year'].between(mdl_forecast_year-11, mdl_forecast_year-1)]
    grain_coef = grain_coef.assign(coef=grain_coef['ah']/grain_coef['ap'])
    grain_coef = grain_coef.groupby(['state_name']).agg(coef=('coef', np.mean)).reset_index()
    grain_coef = grain_coef[['state_name', 'coef']]
    
    ap_df = prod_df[
        (prod_df['statisticcat_desc']=='AREA PLANTED') &
        (prod_df['year']== mdl_forecast_year) &
        (prod_df['reference_period_desc']== 'YEAR') &#'YEAR - MAR ACREAGE') & ##NEED TO COME UP WITH An IF ALGORITHM HERE TO ACCOUNT FOR PRE AND POST 2012
        (prod_df['state_name']!='US TOTAL')
        ]
    ap_df = ap_df.pivot_table(index=['year', 'state_name'], columns='statisticcat_desc', values='Value').reset_index()
    ap_df['state_name'] = ap_df['state_name'].str.title()
    ap_df = ap_df.rename(columns={'AREA HARVESTED':'ah', 'AREA PLANTED':'ap', 'YIELD':'yld'})
    ap_df = ap_df.merge(grain_coef, on=['state_name'], how='left').dropna()
    ap_df = ap_df.assign(value = ap_df['ap']*ap_df['coef'])
    
    wt_df = ap_df.assign(wt_value = ap_df['value']/ap_df['value'].sum())
    wt_df = wt_df[['state_name', 'year', 'wt_value']]
    wt_df['state_name'] = wt_df['state_name'].str.title() # to make compatible with wx and pred df merges
    
    print('State-area weight calculations complete')
    
    # Aggregated weighted mid-point estimate
    
    mid_pred = state_yhat_df.merge(wt_df, on='state_name', how='left')
    mid_fcast = mid_pred.assign(wt_yhat = mid_pred['yld_hat']*mid_pred['wt_value'])
    state_mid = mid_fcast
    mid_fcast = np.sum(mid_fcast['wt_yhat'])
    print('Mid-point forecast: ', mid_fcast.astype('float64'))
    
    # Develop nonparametric stochastic weather simulations
    
    year_list = np.arange(weather_draw_start_year, mdl_forecast_year-1)
    np.random.seed(5) # set's the seed for reproducible results
    year_draw = np.random.choice(year_list, num_of_draws, replace=True)
    
    wx_draw = pd.DataFrame()
    
    draw_count = 0
    
    for y in year_draw:
        y_df = wx_mdl[wx_mdl['year']==y]
        draw_count += 1
        y_df=y_df.assign(draw_count=draw_count)
        wx_draw = pd.concat([wx_draw, y_df])
        print(f'Draw for model year {mdl_forecast_year}: {draw_count}/{num_of_draws}') # include this in the code to monitor the progress of the draws (takes about 1 minute per 1,000 draws)
    
    wx_draw = wx_draw.reset_index(drop=True).rename(columns={'year':'wx_year'})
    
    nonparam_df = wx_draw.merge(wt_df[['state_name', 'wt_value']], on='state_name', how='left', suffixes=['', '_wt']).sort_values('draw_count').reset_index(drop=True)
    
    plt_adj_norm_dict = {'state_name':state_list, 'plt_adj':[1]*len(state_list)}
    plt_adj_norm= pd.DataFrame.from_dict(plt_adj_norm_dict)
    
    current_prog_df = prog_mdl[(prog_mdl['year']==mdl_forecast_year)&
                               (prog_mdl['month']==plt_prog_month)&
                               (prog_mdl['day']==plt_prog_day)]
    plt_adj_current_dict = {'state_name':state_list} 
    plt_adj_current = pd.DataFrame.from_dict(plt_adj_current_dict).merge(current_prog_df, on='state_name', how='left')
    
    
    if plt_prog_toggle == False:
        state_plt_pred = plt_adj_norm
    elif prog_mdl['week_ending'].max() >= pd.to_datetime(str(mdl_forecast_year) + '-' + str(plt_prog_month) + '-' + str(plt_prog_day)):
        state_plt_pred = plt_adj_current
    else:
        state_plt_pred = plt_adj_norm
    
    
    nonparam_df = nonparam_df.merge(state_plt_pred, on='state_name', how='left').rename(columns={'year':'wx_year'})
    nonparam_df['year'] = mdl_forecast_year
    
    nonparam_results  = pd.DataFrame()
    
    nonparam_state_list = state_yhat_df['state_name'].unique()
    
    for s in nonparam_state_list:
        mdl_s = state_mdl_dict[s][2]
        df = nonparam_df[nonparam_df['state_name']==s]
        results = df.assign(yhat = mdl_s.predict(df))
        results = df.assign(wt_yhat = results['yhat'] * results['wt_value'])
        nonparam_results = pd.concat([nonparam_results, results])
        print('Success: ' + str(s))
    
    nonparam_national_results = nonparam_results.groupby(['draw_count']).agg({'wt_yhat':np.sum})
    print('Weather data random draw and merge complete')
    
    # Distribution of stochastic results
    
    fig, ax = plt.subplots()
    sns.histplot(nonparam_national_results['wt_yhat'], kde=True, bins=30)
    plt.axvline(x=mid_fcast, ymax=1, color='red', linestyle='dotted', label = 'midpoint')
    plt.axvline(x=nonparam_national_results['wt_yhat'].median(), color='green', linestyle='dotted', label ='median')
    plt.axvspan(nonparam_national_results['wt_yhat'].quantile(.25), nonparam_national_results['wt_yhat'].quantile(.75), alpha=0.2, color = 'green')
    plt.axvspan(nonparam_national_results['wt_yhat'].quantile(.1), nonparam_national_results['wt_yhat'].quantile(.9), alpha=0.1, color = 'green')
    plt.axvline(x=prod_mdl[(prod_mdl.year==mdl_forecast_year) & (prod_mdl.state_name=='Us Total')]['yld'].squeeze(), color='grey', label = 'final')
    try:
        plt.axvline(x=hist_ylds[hist_ylds.mdl_year==mdl_forecast_year]['June'].squeeze(), color='orange', linestyle='dotted', label = 'June WASDE')
    except:
        pass
    plt.title(f'Stochastic simulation of weather on adjusted trend\nyields for U.S. {mdl_forecast_year} corn crop with {num_of_draws}\nstate-level weather draws since {weather_draw_start_year}')
    plt.xlabel('Bushels per acre')
    plt.ylim(0, 1800)
    ax.legend(loc='upper left')
    plt.savefig('docs/National_corn_stochastic_distribution_'+str(mdl_forecast_year)+'.png')
    plt.close()

    print('Mid point (non-stocastic) for average weather: ', mid_fcast)
    print('Average: ', nonparam_national_results['wt_yhat'].mean())
    print('Median: ', nonparam_national_results['wt_yhat'].median())
    print('IQR: ', nonparam_national_results['wt_yhat'].quantile(.25), ' and ',  nonparam_national_results['wt_yhat'].quantile(.75))
    print('90% confidence interval: ', nonparam_national_results['wt_yhat'].quantile(.1), ' and ',  nonparam_national_results['wt_yhat'].quantile(.9))
    
    stoch_mdl_stats_yr = {
        'model_year': mdl_forecast_year,
        'midpoint':mid_fcast,
        'average': nonparam_national_results['wt_yhat'].mean(),
        'median': nonparam_national_results['wt_yhat'].median(),
        'iqr_lo': nonparam_national_results['wt_yhat'].quantile(.25), 
        'iqr_hi': nonparam_national_results['wt_yhat'].quantile(.75),
        'ci_lo': nonparam_national_results['wt_yhat'].quantile(.1),
        'ci_hi': nonparam_national_results['wt_yhat'].quantile(.9)
        }
    stoch_mdl_stats_dict[str(fc_yr)]=stoch_mdl_stats_yr
    
stoch_mdl_stats_df = pd.DataFrame.from_dict(stoch_mdl_stats_dict, orient='index')

results_comp = stoch_mdl_stats_df.merge(hist_ylds, left_on='model_year', right_on='mdl_year', how='left')
results_comp = results_comp.merge(prod_mdl[prod_mdl.state_name=='Us Total'][['year', 'yld']], left_on='model_year', right_on='year', how='left')
results_comp.to_csv('docs/corn_yield_mdl_results_comp.csv')

plt.plot(results_comp['model_year'], results_comp['midpoint'], c='red', label='midpoint')
plt.plot(results_comp['model_year'], results_comp['median'], c='blue', label='median')
plt.plot(results_comp['model_year'], results_comp['ci_lo'], color='green', alpha=.5, label='90% High')
plt.plot(results_comp['model_year'], results_comp['ci_hi'], color='green', alpha=.5, label='90% Lo')
plt.scatter(x=results_comp['model_year'], y=results_comp['yld'], color = 'grey', label='NASS Final')
plt.scatter(x=results_comp['model_year'], y=results_comp['June'], color = 'orange', label='June WASDE')
plt.legend(loc='lower right')
plt.title('Comparison of Model results versus NASS Final Yield, Corn')
plt.savefig('docs/comp_chart.png')
plt.show()

##Merge with June WASDE and Final data
