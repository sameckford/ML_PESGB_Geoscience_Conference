# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:30:07 2019

@author: Samuel Eckford
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
import os
#plt.style.use('seaborn')
from scipy.interpolate import griddata
import pandas as pd
import matplotlib.cm as cm
from multiprocessing.dummy import Pool as ThreadPool
import scipy.stats as stats

pool = ThreadPool(4)

def Point_Set_Load_and_Merge_Slices(point_set_DBAMP, point_set_TTOP, filename2):
    """Load in a point set from HIIP and merge it with a DTAMP dataframe. This can attach X and Y coorindates to the dataframe for horizon based operations"""
    XY_Dataframe_DBAMP = pd.read_csv(point_set_DBAMP, delim_whitespace=True, names=['X', 'Y', 'DBAMP'], skiprows=2)
    XY_Dataframe_TTOP = pd.read_csv(point_set_TTOP, delim_whitespace=True, names=['X', 'Y', 'TTOP'], skiprows=2)
    print(XY_Dataframe_DBAMP.shape)
    print(XY_Dataframe_TTOP.shape)
    XY_Dataframe_DBAMP = XY_Dataframe_DBAMP.round(2)
    XY_Dataframe_TTOP = XY_Dataframe_TTOP.round(2)
    Merged_XY_Dataframe = XY_Dataframe_DBAMP.merge(XY_Dataframe_TTOP, how='left', on=['X', 'Y'])
    today = dt.datetime.today().strftime('%m_%d_%Y')
    filename1 = filename2
    Merged_XY_Dataframe.to_csv("{}_{}.txt".format(filename1, today), sep=" ")
    print(Merged_XY_Dataframe.shape)
    return Merged_XY_Dataframe

def Amplitude_Conformance_Error(amp_df, contact, amp_polarity=1, amp_slices=20, amp_manual2=False):
    """Paired with contact slicing function. For a single contact calculate the F1 score for amplitude slicing.\
    Polarity is 1 for positive and -1 for negative. Takes a dataframe containing DBAMP and TTOP. 20 amp slices by default"""
    DBAMP_TTOP = amp_df[['DBAMP', 'TTOP', 'X', 'Y']].copy()
    #TTOP_min = df['TTOP'].min()
    #TTOP_max = df['TTOP'].max()
    DBAMP_TTOP['DBAMP'] = DBAMP_TTOP['DBAMP'] * amp_polarity
    DBAMP_Mask = DBAMP_TTOP['DBAMP'] >= 0
    DBAMP_TTOP = DBAMP_TTOP[DBAMP_Mask]
    contact = float(contact)
    if amp_manual2 == False:
        DBAMP_01perc = DBAMP_TTOP['DBAMP'].quantile(.01)
    else:
        DBAMP_01perc=amp_manual2[1]
    if amp_manual2 == False:
        DBAMP_99perc = DBAMP_TTOP['DBAMP'].quantile(.99)
    else:
        DBAMP_99perc = amp_manual2[0]
    amp_cutoff = np.linspace(DBAMP_01perc, DBAMP_99perc, num=amp_slices)
    f1_dataframe = pd.DataFrame()

    for amp in amp_cutoff:
        #print("amplitude is %s" % amp)
        DBAMP_TTOP['Amp_Conditional'] = np.where((DBAMP_TTOP['DBAMP'] >= amp) & (DBAMP_TTOP['TTOP'] <= contact), 'a', \
                  np.where((DBAMP_TTOP['DBAMP'] >= amp) & (DBAMP_TTOP['TTOP'] >= contact),'b', \
                  np.where((DBAMP_TTOP['DBAMP'] <= amp) & (DBAMP_TTOP['TTOP'] <= contact), 'c', 'd')))
        tp =  "a"
        tpc = DBAMP_TTOP.Amp_Conditional.str.count(tp).sum()
        #print(tpc)
        fn =  "b"
        fnc = DBAMP_TTOP.Amp_Conditional.str.count(fn).sum()
        fp =  "c"
        fpc = DBAMP_TTOP.Amp_Conditional.str.count(fp).sum()
        tn =  "d"
        tnc = DBAMP_TTOP.Amp_Conditional.str.count(tn).sum()
        recall = tpc / (tpc+fnc)
        #print("Recall = %s" % recall)
        precision = tpc / (tpc+fpc)
        #print("Precision = %s" % precision)
        f1 = 2 * (recall * precision) / (recall + precision)
        #print("F1 score = %s" % f1)
        temp_list = [amp, f1, contact]
        temp = pd.DataFrame({'amplitude': [amp], 'f1_score_(%)': [f1], 'contact_(m)': [contact]})
        f1_dataframe = pd.concat([f1_dataframe, temp])
        #oldpath = os.getcwd()
        #new_path = "\\contacts"
        #complete_path = oldpath + new_path
        #if not os.path.exists(complete_path):
        #    os.makedirs(complete_path)
        #f1_dataframe.to_csv(r"Contact_%s_amplitude_error.txt" % (contact), sep=" ")
    f1_dataframe.reset_index(inplace=True, drop=True)
    print(f1_dataframe.shape)
    print(f1_dataframe.info())
    return f1_dataframe


def Contact_Slicing(contact_df, slice_range='culmination', slice_bottom='deepest', contact_slices=20, polarity=1, amp_manual=False, vertical_lines=False, contact_filename=False, contact_counter=False):
    """Iterative slices of contacts through a structure which can then be fed into an amplitude conformance function for complex conformance solving.
    Takes a Dataframe containing amplitude (labelled DBAMP) and depth (labelled TTOP).
    pick the slice_range as either 'culmination' or 'offset' ('offset' by default), this will slice contacts from the bottom slice to either the culmination,
    or it will ask for an offset value.
    slice_top is the deepest contact slice. It is set to 'deepest' by default which means the function willl automatically define
    the bottom slice from the deepest depth point. Otherwise if a number is present it will define it through this value entered.
    contact_slices allows the number of slices ot be selected
    Polarity = 1 for positive impedance of -1 for negative. It is positive by default
    amp_manual=False by default but a manual amplitude range can be input as a list e.g. [0, 2000] representing the minimum and maximum"""
    if slice_bottom=='deepest':
        TTOP_max = contact_df['TTOP'].max()
    else:
        TTOP_max = slice_bottom
    if slice_range=='culmination':
        TTOP_min = contact_df['TTOP'].min()
    elif slice_range == 'offset':
        TTOP_min = TTOP_max - float(input('Enter contact offset: '))
    else:
        TTOP_min = slice_range
    #TTOP_min = TTOP_max - slice_range
    contact_slice_array = np.linspace(TTOP_min, TTOP_max, num=contact_slices)
    temp_contact = pd.DataFrame(columns= ['amplitude', 'f1_score_(%)', 'contact_(m)'])
    #Plot formatting and styling
    #plt.style.use('classic')
    n_lines = contact_slices
    color_array=cm.hsv(np.linspace(0, 1, n_lines))
    #ax = plt.axes()
    #ax.set_color_cycle([plt.cm.cool(i) for i in np.linspace(0, 1, n_lines)])
    #For loop to iterate over each contact slice
    x_lines = []
    sns.set_style("ticks")
    for i, j in zip(contact_slice_array, color_array):
        #Calling the amp error function to iterate over each contact with 20 amp slices
        #slice = Amplitude_Conformance_Error(contact_df, i, amp_polarity = polarity, amp_slices=contact_slices, amp_manual2=amp_manual)
        slice = Amplitude_Conformance_Error(contact_df, i, amp_polarity = polarity, amp_slices=contact_slices, amp_manual2=amp_manual)
        print(slice.head())
        if vertical_lines==True:
            x_vert1 = slice['f1_score_(%)'].idxmax()
            x_vert = slice.iloc[x_vert1, 0]
            #Append each contact amp error info to a master dataframe
            x_lines.append(x_vert)
        #Add a line to the plot for each contact
        ax = sns.lineplot(x=slice['amplitude'], y=slice['f1_score_(%)'], alpha=0.4, label='{}_(m)'.format(i), c=j)
                    
        temp_contact = temp_contact.append(slice)
    temp_contact.reset_index(inplace=True, drop=True)
    #Plot vertical line if vertical_lines=True
    if vertical_lines ==True:
        for v in x_lines:
            ax = plt.axvline(x=v, alpha=0.2, linestyle=':')
        #Plot rectangle spanning the vertical lines
        plt.axvspan(min(x_lines), max(x_lines), color='hotpink', alpha=0.05)
    #Plot asthetics
    plt.title=('Amplitude vs. f1 score for contact slices')
    plt.xlabel=('amplitude')
    plt.ylabel=('f1_score')
    plt.grid()
    leg = plt.legend(['%0.1f' % v for v in contact_slice_array], ncol=2, prop={'size': 6}, fontsize=10)
    leg.set_title('contact_(m)')
    #plt.legend()
    plotname = "{}_{}".format(contact_filename, contact_counter)
    #plotname = dt.datetime.now()
    #Save the figure
    plt.savefig('%s.png' % (plotname), bbox_inches='tight',dpi=200)
    plt.show()
    return temp_contact, x_lines

def Amplitude_conformance_meshgrid(amp1_df, amp_cutoff, contact, polarity=1, meshgrid_filename=False):
    #%matplotlib inline
    plt.style.use('seaborn-white')

    #Calculate X and Y bounds
    x_min = amp1_df['X'].min()
    x_max = amp1_df['X'].max()
    y_min = amp1_df['Y'].min()
    y_max = amp1_df['Y'].max()
    
    #Flip polarity
    amp1_df['DBAMP'] = amp1_df['DBAMP'] * polarity

    #Screen DBAMP and depth info with masks
    amp1_df['DBAMP'].mask(amp1_df['DBAMP'] <= amp_cutoff, other=-999, inplace=True)
 
    #Calculate the average X increment
    #amp1_df = amp1_df.sort_values(by=['X'])
    #amp1_df['x_inc'] = amp1_df['X'].shift(-1) - amp1_df['X'] 
    #x_increment = amp1_df['x_inc'].mean()
    #print(x_increment)
    #Calculate the average Y increment
    #amp1_df = amp1_df.sort_values(by=['Y'])
    #amp1_df['y_inc'] = amp1_df['Y'].shift(-1) - amp1_df['Y']
    #y_increment = amp1_df['x_inc'].mean()
    
 
    x_inc_num = round(((x_max - x_min) / 50), 0)
    y_inc_num = round(((y_max - y_min) / 50), 0)
    x = np.linspace(x_min, x_max, x_inc_num)
    y = np.linspace(y_min, y_max, y_inc_num)

    #Create meshgrid
    X, Y = np.meshgrid(x, y)

    #Interpolate Z values
    Z = griddata((amp1_df['X'], amp1_df['Y']), amp1_df['TTOP'], (X, Y), method='cubic')
    print(Z)
    
    Z1 = griddata((amp1_df['X'], amp1_df['Y']), amp1_df['DBAMP'], (X, Y), method='cubic')
    print(Z1)

    z_min = amp1_df['TTOP'].min()
    z_max = amp1_df['TTOP'].max()
    z1_min = amp1_df['DBAMP'].min()
    z1_max = amp1_df['DBAMP'].max()

    
    #plt.contourf(X, Y, Z, 20, cmap='Greys_r')
    
    contours = plt.contour(X, Y, Z, 10, colors='black')
    plt.clabel(contours, inline=True, fontsize=8)

    plt.imshow(Z1, extent=[x_min, x_max, y_min, y_max], origin='lower',\
           cmap='hot', vmin = z1_min, vmax=z1_max, alpha=0.5)

    plt.colorbar()
    
    plotname = "{}_Contour_Overlay".format(meshgrid_filename)
    #plotname = dt.datetime.now()
    #Save the figure
    plt.savefig('%s.png' % (plotname), bbox_inches='tight',dpi=200)
    
    
    plt.show()

def Amplitude_Conformance_Calculator(main_df, TTOP_sep=False, TTOP_df='None', slices=20, seismic_polarity=1, amp_selection='median', filename=input('Enter a file name: ')):
    """Calculates amplitude conformance. Performs an initial pass based on a wide range of contacts.
    After analysing the results it then narrows down the contact range and repeats the process again.
    The output is a range of contacts."""
    #Checks to see if the input data is one dataframe or two sets of point sets.
    if TTOP_sep == True:
        main_df = Point_Set_Load_and_Merge_Slices(main_df, TTOP_df, filename2=filename)
    
    counter = 1
    #First pass of analysis based on a wide range of contacts
    first_pass, xlines = Contact_Slicing(main_df, contact_slices=slices, polarity=seismic_polarity, contact_filename=filename, contact_counter=counter)

    first_pass_pivot = pd.pivot_table(first_pass, values='f1_score_(%)', columns='contact_(m)', index='amplitude')
    #Filter out bad results
    round(first_pass_pivot, 2)
    list1 = list(first_pass_pivot) 
    amp_min = first_pass['amplitude'].min()
    round(amp_min, 2)
    max_values = first_pass_pivot.max(axis=0)
    idx_max = first_pass_pivot.idxmax(axis=0)
    round(idx_max, 2)
    max_concat = pd.concat([max_values, idx_max], axis=1)
    max_concat.reset_index(inplace=True)
    max_concat.columns = ['contact_(m)', 'max_values', 'idx_max']
    max_mask = (max_concat['max_values'] >= 0.05) & (max_concat['idx_max'] > amp_min)
    contacts_filtered = max_concat[max_mask]
    #Define contact limits
    min_contact = contacts_filtered['contact_(m)'].min()
    max_contact = contacts_filtered['contact_(m)'].max()
    
    counter = 2
    #Run the second pass with the restricted contact depths
    second_pass, xlines2 = Contact_Slicing(main_df, slice_range=min_contact, slice_bottom= max_contact, contact_slices=slices,\
                                  polarity=seismic_polarity, vertical_lines=True, contact_filename=filename, contact_counter=counter)
    
    #Second pass pivot table and max f1 and amp scores
    second_pass_pivot = pd.pivot_table(second_pass, index='amplitude', columns='contact_(m)', values='f1_score_(%)')
    amp_max2 = second_pass_pivot.idxmax(axis=1)
    f1_max2 = second_pass_pivot.max(axis=1)
    max_concat2 = pd.concat([f1_max2, amp_max2], axis=1)
    max_concat2.reset_index(inplace=True)
    max_concat2.columns = ['amp_max', 'f1_max', 'contact_(m)']
    print(max_concat2)
    #Summary Statistics for Amplitude
    #amp_max2 = max_concat2['amp_max'].max()
    amp_max2 = max(xlines2)
    #amp_min2 = max_concat2['amp_max'].min()
    amp_min2 = min(xlines2)
    amp2_min_max = [amp_max2, amp_min2]
    slices *= 2
    counter = 3
    #Final iteration
    final_pass, xlines3 = Contact_Slicing(main_df, slice_range=min_contact, slice_bottom= max_contact, contact_slices=slices,\
                                  polarity=seismic_polarity, vertical_lines=True, amp_manual=amp2_min_max, contact_filename=filename, contact_counter=counter)
    
    #Pivot table
    final_pass_pivot = pd.pivot_table(final_pass, index='amplitude', columns='contact_(m)', values='f1_score_(%)')
    amp_max3 = final_pass_pivot.idxmax(axis=1)
    f1_max3 = final_pass_pivot.max(axis=1)
    max_concat3 = pd.concat([f1_max3, amp_max3], axis=1)
    max_concat3.reset_index(inplace=True)
    max_concat3.columns = ['amp_max', 'f1_max', 'contact_(m)']
    #Summary Statistics for Amplitude
    amp_max3 = max_concat3['amp_max'].max()
    amp_min3 = max_concat3['amp_max'].min()
    amp3_75pc = np.percentile(max_concat3['amp_max'], 75)
    amp3_25pc = np.percentile(max_concat3['amp_max'], 25)
    amp3_median = np.median(max_concat3['amp_max'])
    print(amp3_median)
    meshgrid1 = Amplitude_conformance_meshgrid(main_df, amp3_75pc, contact=min_contact, polarity=seismic_polarity)
    print(xlines3)
    #Output file
    final_pass_pivot.to_csv("{}_finalpass_values.txt".format(filename), sep=" ")
    #Plot CDF
    # Choose how many bins you want here
    num_bins = 20
    # Use the histogram function to bin the data
    counts, bin_edges = np.histogram(xlines3, bins=num_bins, normed=True)
    # Now find the cdf
    cdf = np.cumsum(counts)
    # And finally plot the cdf
    #plt.plot(bin_edges[1:], cdf)
    ax1 = sns.lineplot(x=bin_edges[1:], y=cdf)
    
    plt.legend(loc='best')
    plt.title = 'Amplitude CDF'
    ax1.set(xlabel='amplitude', ylabel='CDF')
    plt.grid()
    plt.savefig('CDF.png', bbox_inches='tight',dpi=200)
    plt.show()
    return max_concat3, xlines3




os.chdir(r"C:\Users\BigDog\Desktop\Scripts\Amplitude_Conformance\Data\F09")


F09_amp, xlines = Amplitude_Conformance_Calculator("DBAMP___Top_sand_2_Regridded_1.07.txt", TTOP_sep=True, TTOP_df="Time_Top___Top_sand_2_Regridded_1.07.txt", slices=20, seismic_polarity=-1, amp_selection='median')
