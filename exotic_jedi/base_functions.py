#######################
#### ExoTiC - JEDI ####
#######################
# Exoplanet Timeseries Characterisation - JWST Extraction and Diagnostic Investigatior


# Functions for NIRSpec pixels to spectra analysis!


# Authors: Lili Alderson lili.alderson@bristol.ac.uk
#          David Grant david.grant@bristol.ac.uk


# File Created : 8 July 2022
#				 Lili Alderson lili.alderson@bristol.ac.uk

# Last Updated : 18 August 2022 (LA)
#
#                2 August 2022 (LA)
#                 - added dq checker
#                 - fixed NaN related bug
#                 - fixed plotting bug
#
#                27 July 2022 (LA)
#                 - fixing bugs relating to edge cases
#                 - varifying extraction methods give consistent results 


# Imports
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import TwoSlopeNorm

import os
from tqdm import tqdm

from astropy.io import fits
from astropy.timeseries import LombScargle
from astropy.stats import sigma_clipped_stats

from scipy.optimize import curve_fit
from scipy.signal import medfilt
from scipy.interpolate import interp1d
from scipy import interpolate
from scipy import signal

import pickle


###################
# 2D Image Cleaning
###################


def dq_flat_replace(sci_cube, dq_cube, bits_to_mask=[0, 2, 10, 11], window_size=4):
    '''
    # Load in fits file and find locations of pixels with data quality flags that match those in inputed list
    # Any flagged pixels will be replaced with the median value of pixels in a window along that row
    
    # Inputs
    # sci_cube : 3D science array, as in extension [1] of jwst _rateints.fits file
    # dq_cube : 3D array of data quality flags, as in extension [3] of jwst _rateints.fits file
    # bits_to_mask=[0,2,10,11] : data quality flags that require replacement see Table 3, https://jwst-pipeline.readthedocs.io/_/downloads/en/latest/pdf/
    # window_size=4 : how many pixels either side of the flagged pixel to take the median from

    # Outputs
    # sci_cub : cleaned 3D data cube thingy
    '''

    #dq_cube = data_cube[3].data
    
    #sci_cube = data_cube[1].data.copy()

    #np.shape(dq_cube)

    #np.shape(np.where(dq_cube!=0))

    flags_time = np.where(dq_cube!=0)[0] # finding where the pixels have a data quality flag
    flags_y = np.where(dq_cube!=0)[1]
    flags_x = np.where(dq_cube!=0)[2]
    
    counter=0


    for i in tqdm(range(len(flags_time))):
        
        hiti = flags_time[i]
        hity = flags_y[i]
        hitx = flags_x[i]

        binary_sum = dq_cube[flags_time[i] , flags_y[i] , flags_x[i]]

        bit_array = np.flip(np.array(list(np.binary_repr(binary_sum, width=32))).astype(int))
        
        if np.any(bit_array[bits_to_mask] == 1):
            
            # replace !
            
            window = sci_cube[hiti, hity, max(hitx-window_size,0):min(hitx+window_size,sci_cube.shape[2])]

            sci_cube[hiti, hity, hitx] = np.median(window)
            
            counter+=1
            
    print("Replaced ", counter, " pixels")
    
    return(np.nan_to_num(sci_cube))



def construct_spatial_profile(D_S, poly_order=9, outlier_sigma=20, window_size=10):
    
    D_S_clean = []

    counter=0
    
    # Iterate rows.
    col_pixel_idxs = np.arange(D_S.shape[1])
    for row_idx in range(D_S.shape[0]):

        
        #print(row_idx)
        
        D_S_row = np.copy(D_S[row_idx, :])
        D_S_row_mask = np.ones(D_S_row.shape[0])

        while True:
            # Replace flagged pixels with nearby median.
            
            for flag_idx in np.where(D_S_row_mask == 0)[0]:
                #print("doing a thing", D_S_row[flag_idx])
                
                whatever_A = D_S_row.copy()
                
                D_S_row[flag_idx] = np.median(
                    D_S_row[max(0, flag_idx - window_size): 
                            min(D_S_row.shape[0] - 1, flag_idx + window_size + 1)])
                
                
                #print(np.median(
                #    D_S_row[max(0, flag_idx - window_size): 
                #            min(D_S_row.shape[0] - 1, flag_idx + window_size + 1)]))
                
                #print(whatever_A[flag_idx], D_S_row[flag_idx], whatever_A[flag_idx]-D_S_row[flag_idx], (D_S_row-whatever_A))

            # Fit polynomial to column.
            try:
                p_coeff = np.polyfit(col_pixel_idxs, D_S_row, poly_order)
            except np.linalg.LinAlgError as err:
                print('Poly fit error when constructing spatial profile.')
                return None
            p_col = np.polyval(p_coeff, col_pixel_idxs)

            # Check residuals to polynomial fit.
            res_row = D_S_row - p_col
            dev_row = np.abs(res_row) / np.std(res_row)
            max_deviation_idx = np.argmax(dev_row)
            
            #if np.max(D_S_row) == 70000.:
                #plt.figure()
                #plt.title(np.max(dev_row))
                #plt.plot(col_pixel_idxs, p_col)
                #plt.plot(col_pixel_idxs, D_S_row)
                #plt.show()
            
            #if np.median(D_S_row) < background_level:
            #    outlier_sigma = sigma[0]
            #else:
            #    outlier_sigma = sigma[1]
            
            if dev_row[max_deviation_idx] > outlier_sigma:
                # Outlier: mask and repeat poly fitting.
                D_S_row_mask[max_deviation_idx] = 0
                #print("outlier found")
                
                
                
                counter+=1
                continue
            else:
                D_S_clean.append(D_S_row.copy())
                #print("ds_row",np.max(np.abs(D_S_row-D_S[row_idx, :])))
                break
                
    return(np.nan_to_num(np.array(D_S_clean)), counter)




def outliers_through_space(input_data, replace_window=4, search_window=10, poly_order=9, n_sig=20, plot=False):
    '''
    # scans through each row in each image to search for hot and dead pixels
    # uses a window as it scans to get roughly similiar region of detector (i.e., bc of trace) 
    # fits polynomial to that region, and flags outliers to the fit of the poly
    # replaces outliers with median value in the window region
    
    # Inputs
    # input_data : the 3D data array, where axis=0 is through time
    # window_size=30 : how wide of a window the median should be calculated in
    # poly_order=9 : order of polynomial to fit to windowed region of the row
    # n_sig=20 : the number of sigma outlier to hunt for
    # plot=False : whether to plot up the median difference between the before and after stacks

    # Outputs
    # data_to_window : cleaned 3d data cube thingy
    # counts_per_image : list of total number replaced pixels per image
    '''

    data_to_window=input_data.copy()
    
    counts_per_image = []
        
    for int_idx in tqdm(range(np.shape(data_to_window)[0])):
        
        #print(int_idx)
        
        counter=0
        
        for window_start in range(0, input_data.shape[2], search_window):
            window_hits=[]
            new_data = data_to_window[int_idx, :, window_start:min(search_window+window_start, np.shape(data_to_window)[2])]
        
            replacements, counts = construct_spatial_profile(new_data, poly_order, outlier_sigma=n_sig, window_size=replace_window)
          
            data_to_window[int_idx, :, window_start:min(search_window+window_start, np.shape(data_to_window)[2])] = replacements
            
            counter+=counts
            
        counts_per_image.append(counter)
        
    if plot == True:
        median_before = np.median(input_data, axis=0)
        median_after = np.median(data_to_window, axis=0)
        
        plt.figure()
        plt.title("Median Difference Between Raw and Cleaned Data")
        plt.xlabel("$x$ pixel")
        plt.ylabel("$y$ pixel")
        z=median_before-median_after
        try:
            norm = TwoSlopeNorm(vmin=z.min(), vcenter=0, vmax=z.max())
            pc = plt.pcolormesh(z,cmap = 'seismic', norm=norm)
        except:
            pc = plt.pcolormesh(z,cmap = 'seismic')
        plt.colorbar(pc, orientation='horizontal')
        plt.show

            
    return(np.nan_to_num(data_to_window), counts_per_image)       



def outliers_through_time(input_data, window_size=30, n_sig=6, plot=False):
    '''
    # quickly scans for cosmic rays / telegraph pixels in 3D [integrations x time] data by hunting for sigma outliers
    
    # Inputs
    # input_data : the 3D data array, where axis=0 is through time
    # n_sig=5 : the number of sigma outlier to hunt for
    # plot=False : whether to plot out where the cosmic ray hits were found, and an example of how the data has been cleaned
    
    # Outputs
    # new_data : cleaned 3D data cube thingy
    '''

    new_data=input_data.copy()
    list_rays=[]
    total_hits=[]
    hitsi=[]
    hitsx=[]
    hitsy=[]
    
    
    #for window_start in range(input_data.shape[0]-window_size):
    #    window_hits=[]
    #    new_data = data_to_window[window_start:window_size+window_start]
        
    while True:
        
        sigmas = np.std(new_data, axis=0)
        medians = np.median(new_data, axis=0)

        cr_mask = new_data - medians > sigmas * n_sig

        total_hits.append(np.sum(cr_mask))
        #window_hits.append(np.sum(cr_mask))
        #hitsi.append(np.where(cr_mask!=0)[0])
        hitsx.append(np.where(cr_mask!=0)[2])
        hitsy.append(np.where(cr_mask!=0)[1])
        #new_data[cr_mask] = np.broadcast_to(medians, (new_data.shape[0],) + medians.shape)[cr_mask]

        #if plot == True:
        #    plt.figure()
        #    plt.imshow(cr_mask.sum(axis=0))
        #    plt.xlabel("$x$ pixel")
        #    plt.ylabel("$y$ pixel")
        #    plt.show()
        if cr_mask.sum() == 0:
            print()
            print("No more outliers found")
            print("In total", np.sum(total_hits), "outliers were found")
            #print("In this window", np.sum(window_hits), "outliers were found")
            break
        
        for hiti,hity,hitx in zip(*np.where(cr_mask!=0)):

            window = new_data[max(hiti-window_size,0):min(hiti+window_size,new_data.shape[0]), hity, hitx]

            new_data[hiti, hity, hitx] = np.median(window)
        
                
    if plot == True:
        print("Beginning plotting")
        plt.figure()
        plt.xlabel("Integration")
        plt.ylabel("Counts")
        plt.title("Pixels Where Outliers Were Found and Removed")
        plt.plot(0,0,label='before',color='b')
        plt.plot(0,0, color='g', label='after')
        for loop in tqdm(range(len(hitsx))):
            crs_x = hitsx[loop]
            crs_y = hitsy[loop]
            plt.plot(input_data[:, crs_y, crs_x],color='b')
            plt.plot(new_data[:, crs_y, crs_x],color='g')
        plt.legend()
        plt.show()
        
        #print(hitsx, hitsy)
        
    return np.nan_to_num(new_data)




###################
# Trace Finding and Fitting
###################



def gauss(x, x0, sig, k):
    A = (x - x0)**2 / (2*sig**2)
    return k * np.exp(-A)



def identify_trace(im, width_guess, start, end):
    '''
    # fits a gaussian to each column of the image to find the position and width of trace
    # returns array of trace information and an array of the cropped x pixel values
    
    # Inputs
    # im : the image
    # width_guess : how wide the trace looks
    # start : where we want to start extracting in pixels, bc the trace won't extend all the way across
    # end : where we want to end extracting in pixels, although for 395 this will probably be the end of the array
    
    # Outputs
    # fitted_gaussians : array of the curve_fit gaussian output for each column of the image
    # cropped_length : array of the x pixel values used, so we can match up the gaussians to their columns later on
    '''

    fitted_gaussians = []
    
    im_shape = np.shape(im)
    
    pixels_in_column = np.arange(im_shape[0]) # how tall is the thing
    cropped_length = np.arange(start,end) # cropping the array down bc we don't want to deal with the very edges where 
                                          # the trace doesn't extend to 

    for i in range(start,end):

        column = im[:,i]
        
        centre_guess = np.argmax(column)
        amplitude_guess = np.max(column)
        

        try:
            fit_guess = [centre_guess, width_guess, amplitude_guess]

            fpopt, fpcov = curve_fit(gauss, pixels_in_column, column, p0=fit_guess)
            fitted_gaussians.append(fpopt)

        except ValueError as err:
            #self.log.warn("Gaussian fitting failed, nans present for column={}.".format(i))
            fitted_gaussians.append([np.nan, np.nan, np.nan])

        except RuntimeError as err:
            #self.log.warn("Gaussian fitting took to long for column={}.".format(i))
            fitted_gaussians.append([np.nan, np.nan, np.nan])
    
    return(np.array(fitted_gaussians), cropped_length)



def make_poly(params, xvalues):
    '''
    # generates y values based on the output of polyfit for any order polynomial
    
    # Inputs
    # params : output of polyfit for any order
    # xvalues : array of values to create the polynomial, must be same length as final product needs to be
    
    # Outputs
    # poly : y values of the polynomial, to be plotted against xvalues input
    '''

    poly = np.zeros(len(xvalues)) + params[-1] # create array of zeros and then add the constant term to them
    for i in range(len(params)-1): # loop over the params, but not the last one! we did that already
        poly += params[i] * xvalues ** float(len(params)-i-1) # add each term to the poly value so it'll be say the 2nd order term * x**2, but complicated bc counting is hard
    return poly



def get_aperture(im, trace_width_guess, start, end, poly_orders=[2,6], width=3, medflt=11, extrapolate_method=None, continue_value=[20,0], set_to_edge=False):
    '''
    # Takes image and finds trace position and width by fitting gaussians
    # Gives aperture limits by fitting polynomial to width of the gaussians
    
    # Inputs
    # im : 2D integration image
    # trace_width_guess : an initial guess of the width of the trace for the gaussian to fit it 
    # start : where we want to start extracting in pixels, bc the trace won't extend all the way across
    # end : where we want to end extracting in pixels, although for 395 this will probably be the end of the array
    # poly_orders=[2,6] : the order of polynomial we want to fit to the trace position and width
    # width=6 : number of fwhms to extend aperture to
    # medflt=11 : size of the median filter window applied to the width of the trace
    # extrapolate_method=None : how to extend the aperture if it does not reach the edges of the image, None, "flatten" or "continue"
    # continue_value=[20,0] : how many pixels to the left and right to continue the aperture on for if extrapolate_method='continue'
    # set_to_edge=False : whether to default to the edges of the detector if the defined aperture falls off the detector height
    
    # Outputs
    # trace_position : y pixel position of the trace at each column, same length as xarray
    # trace_width : width of trace at each column, same length as xarray
    # upper_aperture : upper edge of the aperture, full length of image
    # lower_aperture : lower edge of the aperture, full length of image
    # upper_ap : upper edge of the aperture, same length as xarray
    # lower_ap : lower edge of the aperture, same length as xarray
    '''

    trace_info, xarray = identify_trace(im, trace_width_guess, start, end) # fit the gaussians to find where the trace is


    trace_positions = np.array(trace_info[:,0])
    trace_widths = np.array(trace_info[:,1])
    nan_mask = np.isfinite(trace_positions)
    
    position_param = np.polyfit(xarray[nan_mask],trace_positions[nan_mask],poly_orders[0]) # fit polynomials to the trace positions and widths 
    width_param = np.polyfit(xarray[nan_mask],medfilt(trace_widths[nan_mask],medflt),poly_orders[1])  # smooth the widths with a median filter
    
    if extrapolate_method == None:
        
        trace_position = make_poly(position_param, xarray) # get the y values for the polynomial, works for any order!
        trace_width = make_poly(width_param, xarray)

        upper_ap = trace_position + width*trace_width # spit out the upper and lower edges of the aperture
        lower_ap = trace_position - width*trace_width # but only across the limits provided
        
        upper_aperture = upper_ap
        lower_aperture = lower_ap
    
    
    if extrapolate_method == 'flatten':
        
        trace_position = make_poly(position_param, xarray) # get the y values for the polynomial, works for any order!
        trace_width = make_poly(width_param, xarray)

        upper_ap = trace_position + width*trace_width # spit out the upper and lower edges of the aperture
        lower_ap = trace_position - width*trace_width # but only across the limits provided

        upper_aperture = np.hstack(( np.ones(start)*(upper_ap[0]) , upper_ap , np.ones(np.shape(im)[1] - end)*(upper_ap[-1]) ))
        lower_aperture = np.hstack(( np.ones(start)*(lower_ap[0]) , lower_ap , np.ones(np.shape(im)[1] - end)*(lower_ap[-1]) ))

    if extrapolate_method == 'continue':
    
        xarray = np.arange(start-continue_value[0],end+continue_value[1]) # np.arange(im.shape[1])
        
        new_start = start-continue_value[0]
        new_end = end+continue_value[1]
        
        trace_position = make_poly(position_param, xarray) # get the y values for the polynomial, works for any order!
        trace_width = make_poly(width_param, xarray)

        upper_ap = trace_position + width*trace_width # spit out the upper and lower edges of the aperture
        lower_ap = trace_position - width*trace_width # but only across the limits provided

        upper_aperture = np.hstack(( np.ones(new_start)*(upper_ap[0]) , upper_ap , np.ones(np.shape(im)[1] - new_end)*(upper_ap[-1]) ))
        lower_aperture = np.hstack(( np.ones(new_start)*(lower_ap[0]) , lower_ap , np.ones(np.shape(im)[1] - new_end)*(lower_ap[-1]) ))
    
    if set_to_edge == False:
        print(upper_aperture)
        if np.max(upper_aperture) >= im.shape[0]:
            raise Exception("Upper aperture extends to "+str(np.max(upper_aperture))+" and falls off the detector. Please reduce the number of FWHMs the aperture extends to.")
        if np.min(lower_aperture) <= 0:
            raise Exception("Lower aperture extends to "+str(np.min(lower_aperture))+" and falls off the detector. Please reduce the number of FWHMs the aperture extends to.")
        if np.min(upper_aperture) <= 0:
            raise Exception("Upper aperture extends to "+str(np.min(upper_aperture))+" and falls off the detector. Please reduce the number of FWHMs the aperture extends to.")
        if np.max(lower_aperture) >= im.shape[0]:
            raise Exception("Lower aperture extends to "+str(np.max(lower_aperture))+" and falls off the detector. Please reduce the number of FWHMs the aperture extends to.")
    
    if set_to_edge == True:
        upper_aperture[upper_aperture >= im.shape[0]-1] = im.shape[0]-1
        lower_aperture[lower_aperture <= 0] = 0
    
    return(trace_position, trace_width, upper_aperture, lower_aperture, upper_ap, lower_ap)





###################
# 1/f Noise & Spectrum Masking
###################




def f_noise_zone(im, upper_ap, lower_ap, ap_buffers=[0,0], plot=False, vmin=0, vmax=4.5, set_to_edge=False):
    '''
    # Creates a mask to use for calculating the 1/f noise
    # Uses the aperture region (+/- a buffer) to define the unilluminated region
    
    # Inputs
    # im : 2D integration image, mostly for shape checking and plotting
    # upper_ap : the upper polynomial of the aperture shape
    # lower_ap : the lower polynomial of the aperture shape
    # ap_buffers=[0,0] : how much of a buffer region to add to the [upper,lower] aperture edges
    # plot=False : plot the regions you've selected
    # vmin=0 : imshow plotting limits
    # vmin=4.5 : imshow plotting limits
    # set_to_edge=False : whether to default to the edges of the detector if the defined aperture falls off the detector height
    
    # Outputs
    # mask : a 2D array same shape as the image in which 0s are the pixels to be used for 1/f noise
    # upper_buffer_region : array same length as image incase region needs to be plotted in the future
    # lower_buffer_region : array same length as image incase region needs to be plotted in the future
    '''


    upper_ap_buffer = ap_buffers[0] # defining some key values here to make everything slightly more legible
    lower_ap_buffer = ap_buffers[1]
    #ap_start = xarray[0]
    #ap_end = xarray[-1]+1
    
    upper_buffer_region = upper_ap+upper_ap_buffer
    lower_buffer_region = lower_ap-lower_ap_buffer
    
    #upper_buffer_region = np.hstack(( np.ones(ap_start)*(upper_ap[0]+upper_ap_buffer) , upper_ap+upper_ap_buffer , np.ones(np.shape(im)[1] - ap_end)*(upper_ap[-1]+upper_ap_buffer) ))
    #lower_buffer_region = np.hstack(( np.ones(ap_start)*(lower_ap[0]-lower_ap_buffer) , lower_ap-lower_ap_buffer , np.ones(np.shape(im)[1] - ap_end)*(lower_ap[-1]-lower_ap_buffer) ))
    
    if set_to_edge == False:
        if np.max(upper_buffer_region) >= im.shape[0]:
            raise Exception("Upper edge of the mask extends to "+str(np.max(upper_buffer_region))+" and falls off the detector. Please reduce the size of the buffer zone.")
        if np.min(lower_buffer_region) <= 0:
            raise Exception("Lower edge of the mask extends to "+str(np.min(lower_buffer_region))+" and falls off the detector. Please reduce the size of the buffer zone.")
        if np.min(upper_buffer_region) <= 0:
            raise Exception("Upper edge of the mask extends to "+str(np.min(upper_buffer_region))+" and falls off the detector. Please reduce the size of the buffer zone.")
        if np.max(lower_buffer_region) >= im.shape[0]:
            raise Exception("Lower edge of the mask extends to "+str(np.max(lower_buffer_region))+" and falls off the detector. Please reduce the size of the buffer zone.")
    
    if set_to_edge == True:

        upper_buffer_region[upper_buffer_region >= im.shape[0]-1] = im.shape[0]-1
        lower_buffer_region[lower_buffer_region <= 0] = 0
    
    
    if plot == True:
        plt.figure()
        plt.title("Aperture and 1/f Noise Region Limits")
        #plt.plot(xarray,trace_position,color='k',ls='--')
        plt.xlabel("$x$ pixel")
        plt.ylabel("$y$ pixel")

        plt.fill_between(np.arange(np.shape(im)[1]), upper_ap, lower_ap, facecolor = 'None',edgecolor='w')
        plt.fill_between(np.arange(np.shape(im)[1]), upper_buffer_region, lower_buffer_region, facecolor = 'None',edgecolor='limegreen')
        
        plt.imshow(np.log10(im),vmin=vmin,vmax=vmax)
        plt.colorbar(label="log(counts)", orientation='horizontal')

        plt.show()
    
    
    int_lower_region = np.ceil(lower_buffer_region) # turn the regions into integers, 
    int_upper_region = np.ceil(upper_buffer_region) # but always round up so we don't accidentally clip an illuminated pixel
    
    mask = np.zeros(np.shape(im))

    for i in range(np.shape(im)[1]):
    
        mask[int(int_lower_region[i]):int(int_upper_region[i]), i] = 1

    if plot == True:
        plt.figure()
        plt.title("1/f Noise Mask")
        plt.xlabel("$x$ pixel")
        plt.ylabel("$y$ pixel")
        plt.imshow(np.log10(np.ma.masked_array(im,mask=mask)), vmin=vmin,vmax=vmax)
        plt.colorbar(label="log(counts)", orientation='horizontal')
        plt.show()
        
    return(mask , upper_buffer_region, lower_buffer_region)



def remove_fnoise(im, mask, plot=False, vmin=0, vmax=4.5):
    '''
    # Removes 1/f from images by "destripping" - subtracting the median of the unilluminated pixels in each column
    
    # Inputs
    # im : 2D integration image
    # mask : 2D binary array same size as im, with 0s for unilluminated region
    # plot=False : plot up the before and after of the image
    # vmin=0 : imshow plotting limits
    # vmin=4.5 : imshow plotting limits
    
    # Outputs
    # clean_im : the destripped 2D integration image
    '''

    masked_im = np.ma.masked_array(im, mask=mask)

    fnoise = np.ones(np.shape(im)) * np.ma.median(masked_im, axis=0)

    clean_im = np.nan_to_num ( im - fnoise )

    if plot == True:
        plt.figure()
        plt.xlabel("$x$ pixel")
        plt.ylabel("$y$ pixel")
        plt.title("Original")
        plt.imshow(np.log10(im),vmin=vmin,vmax=vmax)
        plt.colorbar(label="log(counts)", orientation='horizontal')
        plt.show()

        plt.figure()
        plt.xlabel("$x$ pixel")
        plt.ylabel("$y$ pixel")
        plt.title("Destripped")
        plt.imshow(np.log10(clean_im),vmin=vmin,vmax=vmax)
        plt.colorbar(label="log(counts)", orientation='horizontal')
        plt.show()
        
        plt.figure()
        plt.xlabel("$x$ pixel")
        plt.ylabel("$y$ pixel")
        plt.title("Residuals")
        plt.imshow((clean_im-im))#,vmin=vmin,vmax=vmax)
        plt.colorbar(label="log(counts)", orientation='horizontal')
        plt.show()
        
    return(clean_im)




def make_1f_stack(im_stack, mask):
	'''
    # Returns a stack of 1/f noise cleaned 2D integration images
    
    # Inputs
    # im_stack : a stack of 2D integration images
    # mask : 2D array equal size to the integration, used to hide illuminated area in 1/f cleaning
    
    # Outputs
    # im_stack : the now 1/f noise removed stack
    
    for i in tqdm(range(ints)):

        im_stack[i] = remove_fnoise(im_stack[i], mask) # make a cleaned version of the image
        
    return(im_stack)
    '''




def ints_to_timeseries(im, mask=None):
    '''
    # Stack all the pixels up in a timeseries, taking into account the read time difference between them
    # Can use a mask if you want to hide a certain part of the image (e.g., illuminated, to check 1/f dominance)
    
    # Inputs
    # im : 2D integration image
    # mask=None : 2D array equal size to im, to mask the image if desired
    
    # Outputs
    # read_times : the times each unmasked pixel was read, 10us gaps in columns and a 120us to jump to a new row
    # pixel_counts : the count value of each unmasked pixel
    '''

    time_jump = 10 * 1e-6 # 10 micro second gap between pixels in a column
    
    read_times = []
    pixel_counts = []
    
    if mask is not None:
        im = np.ma.masked_array(im, mask=mask) # mask the array to avoid illuminated pixels
    
    time = 0 # start at time=0 then add on for each iteration
    pixels_in_col, total_cols = np.shape(im)
    
    for j in range(total_cols):
        
        for i in range(pixels_in_col): # save the count and read time for each pixel in the column
            if not(mask[i][j]): # we don't want to save masked pixels, even as NaNs bc future lomb scargle hates that
                read_times.append(time)
                pixel_counts.append(max(im[i][j],0))
            time += time_jump # increase the time by 10us every new pixel in the column
            
        time+= 12 * time_jump # moving to a new column, this takes 120us
    
    return(read_times, pixel_counts)




def check_1f(im_stack, fnoise_mask, stack=True):
    '''
    # Makes a plot comparing the power spectra of a 2D integration before and after 1/f noise cleaning
    # Uses ints_to_timeseries to untangle the images, then a Lomb Scargle periodogram to make plot
    # Can feed a single image, or a complete stack. Stacks will plot the median power spectrum
    
    # Inputs
    # im_stack : either a stack of 2D integration images, or a single 2D integration image
    # mask : 2D array equal size to the integration, used to hide illuminated area in 1/f cleaning
    # stack=True : whether or not im_stack is actually a stack, or a single image
    '''

    frequencies = np.logspace(np.log10(1/(0.22515000000006288*4)),np.log10(1/(10e-6)),100)

    powers = []
    powers_clean = []

    if stack == True:
        ints, rows, cols = im_stack.shape # how many spectra actually are there?
    if stack == False:
        ints = 1

    for i in tqdm(range(ints)):

        im = im_stack[i]
        clean_im = remove_fnoise(im, fnoise_mask) # make a cleaned version of the image

        times,fluxes = ints_to_timeseries(im, mask=fnoise_mask)
        times_clean,fluxes_clean = ints_to_timeseries(clean_im, mask=fnoise_mask) # turn ims into timeseries

        power = LombScargle(times,fluxes).power(frequencies) # lomb that scargle
        powers.append(power)

        power_clean = LombScargle(times_clean,fluxes_clean).power(frequencies) # lomb that scargle: tokyo drift
        powers_clean.append(power_clean)
    
    if stack == True:
        power_clean = np.median(powers_clean,axis=0) # average the power spectra of all the images
        power = np.median(powers,axis=0)

    print(times[-1])    
    
    plt.figure()
    plt.plot(frequencies,power, label='Before')
    plt.plot(frequencies,power_clean, label='After')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Frequency")
    plt.ylabel("Power Spectrum")
    plt.show()



###################
# Extraction Methods
###################




def basic_extraction(im, upper_ap, lower_ap, xarray=None, set_to_edge=False):
    '''
    # extracts the spectra from the aperture region, using intrapixel methods to handle the polynomial non integer-ness
    
    # Inputs 
    # im : 2D integration image
    # upper_ap : the upper polynomial of the aperture shape
    # lower_ap : the lower polynomial of the aperture shape
    # xarray=None : array of x values to extract if not the full image length
    # set_to_edge=False : whether to default to the edges of the detector if the defined aperture falls off the detector height
    
    # Outputs
    # spectrum : 1D stellar spectrum from the image
    '''

    if xarray is not None:
        im = im[:,xarray[0]:xarray[-1]]
        upper_ap = upper_ap[xarray[0]:xarray[-1]]
        lower_ap = lower_ap[xarray[0]:xarray[-1]]
    
    spectrum = np.zeros(np.shape(im)[1])

    if set_to_edge == False:
    
        if np.max(upper_ap) >= im.shape[0]:
            raise Exception("Upper aperture extends to "+str(np.max(upper_ap))+" and falls off the detector. Please change the aperture.")
        if np.min(lower_ap) <= 0.0:
            raise Exception("Lower aperture extends to "+str(np.min(lower_ap))+" and falls off the detector. Please change the aperture.")
        if np.min(upper_ap) <= 0.0:
            raise Exception("Upper aperture extends to "+str(np.min(upper_ap))+" and falls off the detector. Please change the aperture.")
        if np.max(lower_ap) >= im.shape[0]:
            raise Exception("Lower aperture extends to "+str(np.max(lower_ap))+" and falls off the detector. Please change the aperture.")


    if set_to_edge == True:

        upper_ap[upper_ap >= im.shape[0]-1] = im.shape[0]-1
        lower_ap[lower_ap <= 0] = 0


    for i in range(np.shape(im)[1]):
        
        up = upper_ap[i]#-0.5
        lw = lower_ap[i]#-0.5
        
        #print(int(np.ceil(lw)), int(np.floor(up))+1)
        
        spectrum[i] += np.sum(im[int(np.ceil(lw)) : int(np.floor(up))+1, i]) # add up all the "whole" pixels within the int region
        
        #print(int(np.ceil(lw)), int(np.floor(up)), spectrum[i])
        
    return(spectrum)



def intrapixel_extraction(im, upper_ap, lower_ap, xarray=None, set_to_edge=False):
    '''
    # extracts the spectra from the aperture region, using intrapixel methods to handle the polynomial non integer-ness
    
    # Inputs 
    # im : 2D integration image
    # upper_ap : the upper polynomial of the aperture shape
    # lower_ap : the lower polynomial of the aperture shape
    # xarray=None : array of x values to extract if not the full image length
    # set_to_edge=False : whether to default to the edges of the detector if the defined aperture falls off the detector height

    
    # Outputs
    # spectrum : 1D stellar spectrum from the image
    '''

    if xarray is not None:
        im = im[:,xarray[0]:xarray[-1]]
        upper_ap = upper_ap[xarray[0]:xarray[-1]]
        lower_ap = lower_ap[xarray[0]:xarray[-1]]
        
    upper_ap2=upper_ap.copy()
    
    spectrum = np.zeros(np.shape(im)[1])

    if set_to_edge == False:
    
        if np.max(upper_ap) > im.shape[0]-0.5:
            raise Exception("Upper aperture extends to "+str(np.max(upper_ap))+" and falls off the detector. Please change the aperture, or maybe you meant to have set_to_edge=True?")
        if np.min(lower_ap) < 0.0:
            raise Exception("Lower aperture extends to "+str(np.min(lower_ap))+" and falls off the detector. Please change the aperture, or maybe you meant to have set_to_edge=True?")
        if np.min(upper_ap) < 0.0:
            raise Exception("Upper aperture extends to "+str(np.min(upper_ap))+" and falls off the detector. Please change the aperture, or maybe you meant to have set_to_edge=True?")
        if np.max(lower_ap) > im.shape[0]-0.5:
            raise Exception("Lower aperture extends to "+str(np.max(lower_ap))+" and falls off the detector. Please change the aperture, or maybe you meant to have set_to_edge=True?")


    if set_to_edge == True:
        upper_ap[upper_ap >= im.shape[0]-1] = (im.shape[0]-1)        
        lower_ap[lower_ap <= 0] = 0


    for i in range(np.shape(im)[1]):
        
        up = upper_ap[i]-0.5
        lw = lower_ap[i]-0.5
        
        
        spectrum[i] += np.sum(im[int(np.ceil(lw))+1 : int(np.floor(up))+1, i]) # add up all the "whole" pixels within the int region
        
        #print("hello",int(np.ceil(lw))+1, int(np.floor(up))+1)#, spectrum[i])
        
        #print("int pix", int(np.ceil(up)), up%1., int(np.floor(lw))+1, (1.-(lw%1.)))
        
        if int(np.ceil(up)) == 31:
            top_pixel = (im[int(np.ceil(up)), i]) * 1
        else:
            top_pixel = (im[int(np.ceil(up)), i]) * (up%1.) # grab the intra bit of the upper edge (easy)
        
        if int(np.floor(lw))+1 == 0:
            low_pixel = (im[int(np.floor(lw))+1, i]) * 1
        else:
            low_pixel = (im[int(np.floor(lw))+1, i]) * (1.-(lw%1.)) # grab the intra bit of the lower edge (negative numbers here, hard)
        
        spectrum[i] += top_pixel + low_pixel # add those bonuses on
        
        #print(top_pixel, im[int(np.ceil(up)), i] * up%1., im[int(np.ceil(up)), i], up%1.)
        #print(up, lw)
        
    return(spectrum)



###################
# Cross Correlation and Pixel Shifts
###################



def correlator(spec, template, trim_spec=3, high_res_factor=0.01, trim_fit=10, plot=False):
    '''
    # Cross-correlate a spectrum with a template using scipy.signal.correlate() and .correlation_lags()
    # Provides the shift needed in the x direction

    # Inputs
    # spec : the 1D spectrum that needs shifting
    # template : the template spectrum you will be shifting to
    # trim_spec : number of pixels to cutoff spec at each end for cross-correlation
    # high_res_factor : fraction of pixel to interpolate onto
    # trim_fit : number of interpolated pixels to trim parabolic fit by

    # Outputs
    # better_lag : the shift needed to align the spectra
    '''

    # Trim x function, otherwise overlap always greatest at zero shift.
    x = np.copy(spec[trim_spec:-trim_spec])
    y = np.copy(template)
    
    # Interpolate to higher-resolution.
    interp_fx = interpolate.interp1d(np.arange(0, x.shape[0]), x, kind='cubic')
    x_hr = interp_fx(np.arange(0, x.shape[0] - 1, high_res_factor))
    interp_fy = interpolate.interp1d(np.arange(y.shape[0]), y, kind='cubic')
    y_hr = interp_fy(np.arange(0, x.shape[0] - 1, high_res_factor))

    # Level functions required.
    x_hr -= np.linspace(x_hr[0], x_hr[-1], x_hr.shape[0])
    y_hr -= np.linspace(y_hr[0], y_hr[-1], y_hr.shape[0])
    
    # Cross-correlate.
    correlation = signal.correlate(x_hr, y_hr, mode="full")
    lags = signal.correlation_lags(x_hr.size, y_hr.size, mode="full")
    coarse_lag_idx = np.argmax(correlation)
    coarse_lag = lags[coarse_lag_idx] * high_res_factor  # Nearest sub pixel shift.
    
    # Fit parabola.
    trim_lags = lags[coarse_lag_idx - trim_fit:coarse_lag_idx + trim_fit + 1]
    trim_norm_cc = correlation[coarse_lag_idx - trim_fit:coarse_lag_idx + trim_fit + 1]
    trim_norm_cc -= np.min(trim_norm_cc)
    trim_norm_cc /= np.max(trim_norm_cc)
    p_coeffs = np.polyfit(trim_lags, trim_norm_cc, deg=2)
    lag_parab_hr = -p_coeffs[1] / (2 * p_coeffs[0]) * high_res_factor
    if plot:
        plt.figure()
        plt.scatter(trim_lags, trim_norm_cc, c='#000000')
        plt.plot(trim_lags, np.polyval(p_coeffs, trim_lags))
        plt.scatter(lag_parab_hr / high_res_factor, 1.)
        plt.show()

    return lag_parab_hr



def shift_shift(spectrum, shift, method = 'cubic'):
    '''
    # Use cross correlated spectral shifts to regrid the spectra and align them
    
    # Inputs
    # spectrum : 1D stellar spectrum to regrid
    # shift : the calculated spectral shift value
    # method='cubic' : kind of interpolation to do, see scipy.interpolate.interp1d docs
    
    # Outputs
    # new_spec : shifted 1D stellar spectrum
    '''

    x = np.arange(len(spectrum))
    
    shifted_x = x - shift
    
    interp_function = interp1d(x, spectrum, kind=method, fill_value="extrapolate")
    
    new_spec = interp_function(shifted_x)
    
    return(new_spec)



###################
# Wrap it all together
###################



def get_stellar_spectra(data_cube, upper_ap, lower_ap, set_to_edge = True, xarray=None, flat=None, f_mask=None, extract_method="intrapixel", shift=True, interpolate_mode="cubic", trim_spec=[3,1], high_res_factor=[0.01,0.01], trim_fit=[10,10], plot=False):
    '''
    # Gets those spectra! 
    # loops over all the 2D integration images to remove 1/f noise and perform the extraction, 
    # correlates the spectra to the template then deshifts them by interpolating to new grid
    # spits out spectra and their shifts for detrending
    
    
    # Inputs
    # data_cube : the big 3D stac we want to extrac(t)
    # upper_ap : the upper polynomial of the aperture shape
    # lower_ap : the lower polynomial of the aperture shape
    # flat=None : the flat field as obtained by JWST Stage 2. If none, no flat fielding will be performed
    # f_mask : 2D array which will mask illuminated areas for 1/f cleaning. If none, no 1/f correction will be performed
    # fnoise=True : whether to run the 1/f noise correction
    # extract_method='intrapixel' : method used for extraction, default is intrapixel but basic also there
    # xarray=None : array of x values to extract if not the full image length
    # shift=True : whether to deshift the spectra
    # trim_spec=[3,1] : [x,y] number of pixels to cutoff spec at each end for cross-correlation
    # high_res_factor=[0.01,0.01] : [x,y] fraction of pixel to interpolate onto
    # trim_fit=[10,10] : [x,y] number of interpolated pixels to trim parabolic fit by
    # interpolate_mode='cubic' : method used for 1D interpolation
    
    # Outputs
    # all_spectra : array of aligned 1D stellar spectra
    # all_y_collapse : array of aligned 1D collapses in y direction
    # x_shifts : x shift values for detrending
    # y_shifts : y shift values for detrending
    '''

    x_shifts = []
    y_shifts = []
    
    all_spectra = []
    all_y_collapse = []
    
    ints, rows, cols = data_cube.shape # how many spectra actually are there?
    
    print("Running", extract_method, "extraction on", ints, "spectra")
    if flat == None:
        print("No flat fielding is being performed at this time")
    else:
        print("Flat fielding will be performed")
    if f_mask is not None:
        print("1/f noise is being removed")
    else:
        print("No 1/f noise correction is being performed")
    
    unshifted_x = []
    unshifted_y = []
    
    for i in tqdm(range(ints)):
        
        # Divide out flat if one has been specified
        if flat:
            im = data_cube[i] / flat # SOMETHING ABOUT FLATS HERE FIX THIS LATER!!
        else:
            im = data_cube[i]
        
        if f_mask is not None:
            # Remove 1/f using pre-determined mask for unilluminated area
            #print("here!")
            clean_im = remove_fnoise(im, f_mask)
        else:
            clean_im = im
        
        # Collapse the image so we can get y shifts (up down)
        y_collapse = np.sum(clean_im, axis=1)
        
        # Extract spectra using specified method
        if extract_method == 'intrapixel':
            spectrum = intrapixel_extraction(clean_im, upper_ap, lower_ap, xarray)
        if extract_method == 'basic':
            spectrum = basic_extraction(clean_im, upper_ap, lower_ap, xarray) 
        #elif extract_method != 'intrapixel' or 'basic':
        #    raise Exception("Inputted extract_method is not an option! Please use either 'intrapixel' or 'basic'")
        
        #pad_spec = np.pad(spectrum, (50, 50), 'constant')
        #pad_ycol = np.pad(y_collapse, (50, 50), 'constant')
        
        unshifted_x.append(spectrum)
        unshifted_y.append(y_collapse)
        
    if shift == True:
        
        x_template = np.median(unshifted_x, axis=0)
        y_template = np.median(unshifted_y, axis=0)
    
    if plot == True:
        plt.figure()
     
    if shift == True:
        print("Now calculating shifts")
        for i in tqdm(range(ints)):
            # Get those shifts
            
            x_shift = correlator(unshifted_x[i], x_template, trim_spec=trim_spec[0], high_res_factor=high_res_factor[0], trim_fit=trim_fit[0])
            y_shift = correlator(unshifted_y[i], y_template, trim_spec=trim_spec[1], high_res_factor=high_res_factor[1], trim_fit=trim_fit[1])
        
            x_shifts.append(x_shift)
            y_shifts.append(y_shift)
                
            new_spec_x = shift_shift(unshifted_x[i], x_shift-x_shifts[0], method=interpolate_mode)
            new_coll_y = shift_shift(unshifted_y[i], y_shift-y_shifts[0], method=interpolate_mode)
        
            all_spectra.append(new_spec_x)
            all_y_collapse.append(new_coll_y)
                
            if plot == True:
                plt.plot(all_spectra[i])
            
    if shift == False:
        all_spectra = unshifted_x
        all_y_collapse = unshifted_y
        if plot == True:
            for i in range(ints):
                plt.plot(all_spectra[i])
        
    if plot == True:
        plt.xlabel("$x$ Pixel")
        plt.ylabel("Raw Counts")
        plt.show()                    
    return(np.array(all_spectra), np.array(all_y_collapse), np.array(x_shifts), np.array(y_shifts))



###################
# Visualisation
###################



def compare_2d_spectra(clean_spectra, unclean_spectra, wvl, time, time_units="BJD", spectra_limits=[0.8,1], residual_limits=None, figsize=(14,10)):
    '''
    # Make pretty plots comparing 2D spectra at different stages of the extraction process
    # Useful for checking e.g., how much of an effect 1/f noise correction has had
    # Plots standard and normalised 2D spectra for the two inputs, along with a residual plot to highlight any differences

    # Inputs 
    # clean_spectra : 2D array of the "after" spectra over the observation
    # unclean_spectra : 2D array of the "before" spectra over the observation
    # wvl : 1D array of wavelengths that spectra are plotted against (in microns)
    # time : 1D array of times that each spectra correspond to
    # time_units='BJD' : unit of time array, for axes labelling purposes only
    # spectra_limits=[0.8,1] : tuple of [vmin, vmax] for normalised spectra plot
    # residual_limits=None : tuple of [vmin, vmax] for the residual plot, if none selected, plot defaults to min and max value of the residuals
    # figsize=(14,10) : size of figure
    '''


    X, Y = np.meshgrid(wvl, time)

    Z = (clean_spectra)
    Z_norm = (clean_spectra/ np.median(clean_spectra, axis=0))

    Z_old = (unclean_spectra)
    Z_old_norm = (unclean_spectra/ np.median(unclean_spectra, axis=0))

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(3, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[0,1])
    ax4 = fig.add_subplot(gs[1,1])
    ax5 = fig.add_subplot(gs[2,:])

    im1 = ax1.pcolor(X,Y,Z_old,  cmap = 'Reds_r')
    fig.colorbar(im1, label = 'Flux', ax=ax1)
    ax1.set_ylabel(('Time ('+time_units+')'))
    ax1.set_xlabel('Wavelength ($\mu$m)')

    im2 = ax2.pcolor(X,Y,Z_old_norm,  cmap = 'Reds_r', vmax=spectra_limits[1], vmin=spectra_limits[0])
    ax2.set_ylabel(('Time ('+time_units+')'))
    ax2.set_xlabel('Wavelength ($\mu$m)')
    fig.colorbar(im2, ax=ax2, label = 'Relative flux', format="%.2f")

    im3 = ax3.pcolor(X,Y,Z,  cmap = 'Blues_r')
    fig.colorbar(im3, label = 'Flux', ax=ax3)
    ax3.set_ylabel(('Time ('+time_units+')'))
    ax3.set_xlabel('Wavelength ($\mu$m)')

    im4 = ax4.pcolor(X,Y,Z_norm,  cmap = 'Blues_r', vmax=spectra_limits[1], vmin=spectra_limits[0])
    ax4.set_ylabel(('Time ('+time_units+')'))
    ax4.set_xlabel('Wavelength ($\mu$m)')
    fig.colorbar(im4, ax=ax4, label = 'Relative flux', format="%.2f")
    
    
    residuals = (Z_norm - Z_old_norm)
    if residual_limits == None:
        residual_limits=[np.min(residuals), np.max(residuals)]
    im5 = ax5.pcolor(X,Y,residuals,  cmap = 'Purples_r', vmax=residual_limits[1], vmin=residual_limits[0])
    fig.colorbar(im5, label = 'Residual', ax=ax5, orientation='horizontal')
    ax5.set_ylabel(('Time ('+time_units+')'))
    ax5.set_xlabel(r'Wavelength ($\mu$m)')
    
    plt.tight_layout()

    plt.show()

