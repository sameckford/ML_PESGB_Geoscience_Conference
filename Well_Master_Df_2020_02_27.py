# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:26:47 2019

@author: seckford

Depth Conversion using different functions: Constant Velocity, Depth vs Time, Velocity vs Time and V0+Kz
These are calculated using both well and seismic velocities.
For larger datasets with over 20 values data is split into test and train data
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

os.chdir(r'C:\Users\BigDog\Desktop\Scripts\Depth_Conversion\Data')
# Import well tops dataframe, columns should be the format - ['Well_Name', 'Surface', 'X', 'Y', 'Z', 'MD', 'TWT_Picked', 'Z_depth', 'Vint', 'TWT_Auto']
tops_columns = ['Well_Name', 'Surface', 'X', 'Y', 'Z', 'MD', 'TWT_Picked', 'Z_depth', 'Vint', 'TWT_Auto']
well_master_df = pd.read_csv('22_26b_Well_df_edit.txt', skiprows=1, names=tops_columns)
print(well_master_df.head())
# Select the initial well top to depth convert
print('Number of wells per surface')
nunique_df = well_master_df.groupby('Surface')['X'].nunique()
print(nunique_df)
initial_surface = input('Select the top surface to depth convert. Usually this would be the Seabed: ')
# Create a df for just the Top Surface (Usually Seabed unless only 1 layer dc) for both well and seismic velocities (is = initial surface)
is_df_well = well_master_df.loc[well_master_df['Surface'] == initial_surface].copy()
is_df_seismic = well_master_df.loc[well_master_df['Surface'] == initial_surface].copy()
# Create expanded df with attributes for dc, function below creates this table by using either 'TWT_Auto' or 'TWT_Picked'
def df_table_complete(is_df, surface=initial_surface, Tops='TWT_Auto'):
    is_df_copy = is_df.loc[is_df['Surface'] == surface].copy()
    is_df_copy['OWT_int'] = is_df_copy[Tops] / 2000
    is_df_copy['Start_Z'] = 0
    is_df_copy['mid-point_Z'] = is_df_copy['Z_depth'] / 2
    is_df_copy['Vint'] = is_df_copy['Z_depth'] / is_df_copy[Tops] * 2000
    is_df_copy['Vint_int'] = is_df_copy['Z_depth'] / is_df_copy['OWT_int']
    return is_df_copy
is_df_well = df_table_complete(well_master_df)
is_df_seismic = df_table_complete(well_master_df, Tops='TWT_Picked')
print(is_df_seismic)
print(is_df_seismic.columns)
# Fit regressions for functions (well velocities first)
# Vint
Vint_av_well = is_df_well['Vint'].mean()
# Depth vs Time Fitting
def func_fitting(X_series, y_series, df):
    X_temp = np.array(df[X_series]).reshape(len(is_df_well),1)
    y_temp = np.array(df[y_series]).reshape(len(is_df_well),1)
    regressor_temp = LinearRegression()
    regressor_temp.fit(X_temp, y_temp)
    slope_temp = float(regressor_temp.coef_)
    intercept_temp = float(regressor_temp.intercept_)
    if len(is_df_well['Surface']) > 20:
        Xtemp_train, Xtemp_test, ytemp_train, ytemp_test = train_test_split(X_temp, y_temp, test_size = 1/5, random_state=0)
        return Xtemp_train, Xtemp_test, ytemp_train, ytemp_test, regressor_temp, slope_temp, intercept_temp, X_temp, y_temp
    return regressor_temp, slope_temp, intercept_temp, X_temp, y_temp
if len(is_df_well['Surface']) > 20:
    Xdvt_train, Xdvt_test, ydvt_train, ydvt_test, regressor_dvt, slope_dvt, intercept_dvt, Xdvt_well, ydvt_well\
        = func_fitting('TWT_Auto', 'Z_depth', is_df_well)
else:
    regressor_dvt, slope_dvt, intercept_dvt, Xdvt_well, ydvt_well = func_fitting('TWT_Auto', 'Z_depth', is_df_well)
print(str(regressor_dvt) + str(slope_dvt) + str(intercept_dvt))
# Velocity vs Time Fitting
if len(is_df_well['Surface']) > 20:
    Xvvt_train, Xvvt_test, yvvt_train, yvvt_test, regressor_vvt, slope_vvt, intercept_vvt, Xvvt_well, yvvt_well\
        = func_fitting('TWT_Auto', 'Vint', is_df_well)
else:
    regressor_vvt, slope_vvt, intercept_vvt, Xvvt_well, yvvt_well = func_fitting('TWT_Auto', 'Vint', is_df_well)
# V0+Kz Fitting
if len(is_df_well['Surface']) > 20:
    Xv0k_train, Xv0k_test, yv0k_train, yv0k_test, regressor_v0k, slope_v0k, intercept_v0k, Xv0k_well, yv0k_well\
        = func_fitting('mid-point_Z', 'Vint_int', is_df_well)
else:
    regressor_v0k, slope_v0k, intercept_v0k, Xv0k_well, yv0k_well = func_fitting('mid-point_Z', 'Vint_int', is_df_well)
# Fit Regressions for seismic velocities
# Vint (Seismic)
Vint_av_seismic = is_df_seismic['Vint'].mean()
print(Vint_av_seismic)
if len(is_df_well['Surface']) > 20:
    Xdvt_train_s, Xdvt_test_s, ydvt_train_s, ydvt_test_s, regressor_dvt_s, slope_dvt_s, intercept_dvt_s, Xdvt_seismic, ydvt_seismic\
        = func_fitting('TWT_Picked', 'Z_depth', is_df_seismic)
else:
    regressor_dvt_s, slope_dvt_s, intercept_dvt_s, Xdvt_seismic, ydvt_seismic = func_fitting('TWT_Picked', 'Z_depth', is_df_seismic)
print(str(regressor_dvt_s) + str(slope_dvt_s) + str(intercept_dvt_s))
# Velocity vs Time Fitting
if len(is_df_well['Surface']) > 20:
    Xvvt_train_s, Xvvt_test_s, yvvt_train_s, yvvt_test_s, regressor_vvt_s, slope_vvt_s, intercept_vvt_s, Xvvt_seisimc, yvvt_seismic\
        = func_fitting('TWT_Picked', 'Vint', is_df_seismic)
else:
    regressor_vvt_s, slope_vvt_s, intercept_vvt_s, Xvvt_seismic, yvvt_seismic = func_fitting('TWT_Picked', 'Vint', is_df_seismic)
print(str(regressor_vvt_s) + str(slope_vvt_s) + str(intercept_vvt_s))
# V0+Kz Fitting
if len(is_df_seismic['Surface']) > 20:
    Xv0k_train_s, Xv0k_test_s, yv0k_train_s, yv0k_test_s, regressor_v0k_s, slope_v0k_s, intercept_v0k_s, Xv0k_seismic, yv0k_seismic\
        = func_fitting('mid-point_Z', 'Vint_int', is_df_seismic)
else:
    regressor_v0k_s, slope_v0k_s, intercept_v0k_s, Xv0k_seismic, yv0k_seismic = func_fitting('mid-point_Z', 'Vint_int', is_df_seismic)
print(slope_v0k_s)
print(intercept_v0k_s)
# Create Residuals Table
def fit_predictor(pred, x):
    return (pred.predict(np.array([x]).reshape(1, -1))[0]).astype(float)
def V0Kz_conversion(i, j, k, p):
    return (i * np.exp(p * j)) + (k * ((np.exp(p * j)-1))) / p
def DC_Residual(df, Vint_temp, dvt_r, vvt_r, v0k_slope, v0k_intercept, Tops='TWT_Auto'):
    df['Vint_Z'] = df[Tops]  * Vint_temp / 2000
    df['Vint_Res'] = df['Z_depth'] - df['Vint_Z']
    df['DvT_Z'] = np.vectorize(fit_predictor)(dvt_r, df[Tops])
    df['DvT_Res'] = df['Z_depth'] - df['DvT_Z']
    df['V0+Kz_Z'] = np.vectorize(V0Kz_conversion)(df['Start_Z'], df['OWT_int'], v0k_intercept, v0k_slope)
    df['V0+Kz_Res'] = df['Z_depth'] - df['V0+Kz_Z']
    df['VvT_Z'] = np.vectorize(fit_predictor)(vvt_r, df[Tops]) * df[Tops] / 2000
    df['VvT_Res'] = df['Z_depth'] - df['VvT_Z']
    return df
#Seismic Residuals
is_df_well = DC_Residual(is_df_well, Vint_temp=Vint_av_well, dvt_r=regressor_dvt, vvt_r=regressor_vvt, v0k_slope=slope_v0k, v0k_intercept=intercept_v0k, Tops='TWT_Auto')
is_df_seismic = DC_Residual(is_df_seismic, Vint_temp=Vint_av_seismic, dvt_r=regressor_dvt_s, vvt_r=regressor_vvt_s, v0k_slope=slope_v0k_s, v0k_intercept=intercept_v0k_s, Tops='TWT_Picked')
print(is_df_seismic)
print(is_df_seismic['Vint_Res'].mean())
print(is_df_seismic['Vint_Res'].std())

#Plot the DvT relationship
ax1 = plt.subplot(3, 2, 1)
ax1 = plt.scatter(Xdvt_well, ydvt_well, color='red')
ax1 = plt.plot(Xdvt_well, regressor_dvt.predict(Xdvt_well), color='blue')
ax1 = plt.title("Depth vs Time (Well)")
ax1 = plt.xlabel("TWT (Well)")
ax1 = plt.ylabel("Depth (ft)")
ax1 = plt.gca().invert_yaxis()


#Plot vvt relationship
ax2 = plt.subplot(3, 2, 3)
ax2 = plt.scatter(Xvvt_well, yvvt_well, color='red')
ax2 = plt.plot(Xvvt_well, regressor_vvt.predict(Xvvt_well), color='blue')
ax2 = plt.title("Velocity vs Time (Well)")
ax2 = plt.xlabel("TWT (Well)")
ax2 = plt.ylabel("Vint (ft/s)")
#fig= plt.figure(constrained_layout=True)
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig = plt.figure(constrained_layout=True)
fig.set_figheight(15)
fig.set_figwidth(15)
plt.show()

plt.subplot(3, 2, 1)
plt.scatter(Xdvt_well, ydvt_well, color='red')
plt.plot(Xdvt_well, regressor_dvt.predict(Xdvt_well), color='blue')
plt.title("Depth vs Time (Well)")
plt.xlabel("TWT (Well)")
plt.ylabel("Depth (ft)")
plt.gca().invert_yaxis()

plt.subplot(3, 2, 3)
plt.scatter(Xvvt_well, yvvt_well, color='red')
plt.plot(Xvvt_well, regressor_vvt.predict(Xvvt_well), color='blue')
plt.title("Velocity vs Time (Well)")
plt.xlabel("TWT (Well)")
plt.ylabel("Vint (ft/s)")
#Plot v0k relationship
plt.subplot(3, 2, 5)
plt.scatter(Xv0k_well, yv0k_well, color='red')
plt.plot(Xv0k_well, regressor_v0k.predict(Xv0k_well), color='blue')
plt.title("V0 + Kz (Well)")
plt.xlabel("Mid-Point Depth (ft)")
plt.ylabel("Vint_int (Well ft/s)")
plt.gca().invert_yaxis()
#Plot the DvT relationship
plt.subplot(3, 2, 2)
plt.scatter(Xdvt_well, ydvt_well, color='red')
plt.plot(Xdvt_well, regressor_dvt.predict(Xdvt_well), color='blue')
plt.title("Depth vs Time (Seismic)")
plt.xlabel("TWT (Seismic)")
plt.ylabel("Depth (ft)")
plt.gca().invert_yaxis()
#Plot vvt relationship
plt.subplot(3, 2, 4)
plt.scatter(Xvvt_well, yvvt_well, color='red')
plt.plot(Xvvt_well, regressor_vvt.predict(Xvvt_well), color='blue')
plt.title("Velocity vs Time (Seismic)")
plt.xlabel("TWT (Seismic)")
plt.ylabel("Vint (ft/s)")
#Plot v0k relationship
plt.subplot(3, 2, 6)
plt.scatter(Xv0k_well, yv0k_well, color='red')
plt.plot(Xv0k_well, regressor_v0k.predict(Xv0k_well), color='blue')
plt.title("V0 + Kz (Seismic)")
plt.xlabel("Mid-Point Depth (ft)")
plt.ylabel("Vint_int (Seismic ft/s)")
plt.gca().invert_yaxis()
#Show plots
plt.subplots(constrained_layout=True)
#   plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.show()






    






