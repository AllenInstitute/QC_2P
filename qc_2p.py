# -*- coding: utf-8 -*-
import os
import h5py
import numpy as np
from scipy.linalg import LinAlgError
from scipy.stats import gaussian_kde
import matplotlib.pylab as plt

class Qc2p():
    """QC class for 2p movies

    Objects contain methods to generate QC metrics from physio movies 

    data should be contained in an hdf5 as a TxXxY array where T is number of time frames. 

    Args:  
        filepath:  The full path to the two-photon physio movie 
        h5_path_to_data: internal path used to access the movie data into the hdf5 file
        dyn_range:  An array corresponding to the min and max pixel value

    Attributes: 
        data_pointer:  A reference to the HDF5 file of interest
        _cache:  A cache that stores previously subsampled and cropped videos
    """

    def __init__(self,  filepath, h5_path_to_data = 'data', dyn_range = [0, 65533]):
        h5_pointer = h5py.File(filepath,'r')
        self.data_pointer = h5_pointer[h5_path_to_data]
        self._dyn_range = dyn_range
        self._cache = {}
    
    def validate_crop(self, crop):
        """Helper function to ensure cropping is not too extreme.

        Args:  
            crop:  (px_y, px_x), the number of pixels to remove from the borders
        
        Returns:  
            A possibly adjusted crop tuple.
        """
        px_y, px_x = crop
        if 2*crop[0] > self.data_pointer.shape[1]:
            px_y = 0
        if 2*crop[1] > self.data_pointer.shape[2]:
            px_x = 0
        return (px_y, px_x)

    def subsample_and_crop_video(self, subsample, crop, start_frame=0, end_frame=-1):
        """Subsample and crop a video, cache results. Also functions as a data_pointer load.

        Args: 
            subsample:  An integer specifying the amount of subsampling (1 = full movie)
            crop:  A tuple (px_y, px_x) specifying the number of pixels to remove
            start_frame:  The index of the first desired frame
            end_frame:  The index of the last desired frame

        Returns: 
            The resultant array.
        """
        if (subsample, crop[0], crop[1], start_frame, end_frame) in self._cache:
            return self._cache[(subsample, crop[0], crop[1], start_frame, end_frame)]

        _shape = self.data_pointer.shape
        px_y_start, px_x_start = crop
        px_y_end = _shape[1] - px_y_start
        px_x_end = _shape[2] - px_x_start
        
        if start_frame == _shape[0] - 1 and (end_frame == -1 or end_frame == _shape[0]):
            cropped_video = self.data_pointer[start_frame::subsample, 
                                                px_y_start:px_y_end,
                                                px_x_start:px_x_end]
        else:
            cropped_video = self.data_pointer[start_frame:end_frame:subsample, 
                                                px_y_start:px_y_end,
                                                px_x_start:px_x_end]

        self._cache[(subsample, crop[0], crop[1], start_frame, end_frame)] = cropped_video

        return cropped_video

    def plot_intensity_histogram(self, start_frame=1, end_frame=501, log_scale=True, crop=(10,10)):
        """Obtain a plot of the intensity histogram. 

        Args: 
            start_frame, end_frame:  Integers specifying the beginning and end frame to use
            log_scale:  A boolean specifying whether or not to plot on a log scale
        
        Returns: 
            A figure.
        """
        dynamic_range = self.get_dynamic_range()
        pixel_values = self.subsample_and_crop_video(subsample=1, crop=crop,
                                                     start_frame=start_frame, 
                                                     end_frame=end_frame)

        fig = plt.figure()
        pixel_values = pixel_values.flatten()
        plt.hist(pixel_values, bins=100, range=dynamic_range, normed=True, log=log_scale, histtype='bar', ec='black')
        plt.xlabel("Pixel Value")
        plt.ylabel("pdf")
        return fig

    def plot_poisson_curve(self, start_frame=1, end_frame=2001, crop=(150,150), perc_min=3, perc_max=90):
        """Obtain a plot showing Poisson characteristics of the signal.

        Args: 
            start_frame, end_frame: Integers specifying the beginning and end frame to use
            crop:  A tuple (px_y, px_x) specifying the number of edge pixels to remove
            perc_min, perc_max:  Min and max values between 0-100 used in filtering based on percentile

        Returns: 
            A figure.
        """
        end_frame = min(end_frame, self.data_pointer.shape[0])
        photon_gain = self.get_photon_gain_parameters(start_frame=start_frame, 
                                                      end_frame=end_frame, 
                                                      crop=crop,
                                                      perc_min=perc_min,
                                                      perc_max=perc_max)

        h, xedges, yedges = np.histogram2d(photon_gain['var'], photon_gain['mean'], bins=(200,200))
        extent = [yedges[0], yedges[-1], xedges[0], xedges[-1]]
     
        fig = plt.figure()
        plt.imshow(h, origin='lower', extent=extent, aspect='auto', cmap='Blues')
        plt.colorbar()
        plt.xlabel('Mean')
        plt.ylabel('Variance')
        plt.xlim(photon_gain['mean'].min(), photon_gain['mean'].max())
        plt.ylim(photon_gain['var'].min(), photon_gain['var'].max())
              
        mean_range = np.linspace(0, photon_gain['mean'].max(), num=200)
        background_noise_mean = -photon_gain['offset'] / photon_gain['slope']
        plt.tight_layout()
        plt.plot(mean_range, photon_gain['slope'] * (mean_range - background_noise_mean), 'r')
        return fig

    def get_dynamic_range(self):
        """Get the dynamic range set for the data

        Returns: 
            An array corresponding to the min and max pixel value"""

        return self._dyn_range

    def get_saturation_metrics(self, subsample=1, start_frame=1, end_frame=501, crop=(0,0)):
        """Return the number of pixels falling outside the usual dynamic range.

        Args: 
            subsample:  An integer specifying the amount of subsampling (1 = full movie)
            crop:  A tuple (px_y, px_x) specifying the number of edge pixels to remove
            start_frame:  Index of start frame
            end_frame:  Index of end frame

        Returns: 
            An dictionary of saturation metrics.
        """
        pixel_range = self.get_dynamic_range()
        subsampled_data = self.subsample_and_crop_video(subsample=subsample, 
                                                        crop=crop, 
                                                        end_frame=end_frame,
                                                        start_frame=start_frame)

        max_plane = subsampled_data.max(axis=0).flatten()
        nb_saturated_pixels = int((max_plane >= pixel_range[1]).sum())

        nb_undersat_pixels = (subsampled_data.flatten() <= pixel_range[0]).sum()
        nb_undersat_pixels = nb_undersat_pixels / float(subsampled_data.shape[0])
        nb_undersat_pixels_perc = 100 * nb_undersat_pixels / float(subsampled_data.shape[1] * subsampled_data.shape[2])

        return {'nb_saturated_pixels': nb_saturated_pixels,
                'nb_low_pixels': int(nb_undersat_pixels),
                'nb_low_pixels_perc': nb_undersat_pixels_perc}

    def get_axis_mean(self, axis=2, subsample=1, crop=(0,0), start_frame=0, end_frame=-1):
        """Get the mean of the video across a given axis.

        Args: 
            axis:  The axis (0, 1, 2, or None) over which to average
            subsample:  An integer specifying the amount of subsampling (1 = full movie)
            crop:  A tuple (px_y, px_x) specifying the number of pixels to remove
            start_frame:  The index of the first desired frame
            end_frame:  The index of the last desired frame

        Returns: 
            The averaged array or value, depending on the chosen axis.
        """
        cropped_video = self.subsample_and_crop_video(subsample=subsample, 
                                                      crop=crop, 
                                                      start_frame=start_frame, 
                                                      end_frame=end_frame)
        return np.mean(cropped_video, axis=axis)

    def get_percent_change_intensity(self, subsample=1, crop=(0,0), nb_border_frames_to_avg=200, ignore_frames=300):
        """Return the percent change in total intensity throughout the physio movie.

        Args: 
            subsample:  An integer specifying the amount of subsampling (1 = full movie)
            crop:  A tuple (px_y, px_x) specifying the number of edge pixels to remove
            nb_border_frames_to_avg:  Number of frames included in the endpoint averages - care
                                      should be taken as this is done after subsampling
            ignore_frames:  Number of frames to ignore at the beginning and end of the movie

        Returns: 
            A float corresponding to the loss in total intensity as a percentage.
        """
        start_mean = self.get_axis_mean(axis=(0,1,2), 
                                        subsample=subsample, 
                                        start_frame=ignore_frames,
                                        end_frame=nb_border_frames_to_avg+ignore_frames)
        end_mean = self.get_axis_mean(axis=(0,1,2), 
                                      subsample=subsample, 
                                      start_frame=-(ignore_frames+nb_border_frames_to_avg+1),
                                      end_frame=-ignore_frames)

        percent_change_intensity = 100 * (end_mean - start_mean) / start_mean
        return percent_change_intensity

    def get_snr_metrics(self, start_frame=1, end_frame=2001, crop=(30,30)):
        """SNR Metrics.

        Compute metrics related to the signal to noise level for the images.

        Args:  
            start_frame, end_frame: Integers specifying the beginning and end frame to use
            crop:  A tuple (px_y, px_x) specifying the number of edge pixels to remove
        
        Returns:  
            A dictionary of metrics
        """
        cropped_video = self.subsample_and_crop_video(subsample=1,
                                                      start_frame=start_frame,
                                                      end_frame=end_frame,
                                                      crop=self.validate_crop(crop))

        # Simple SNR - https://en.wikipedia.org/wiki/Signal-to-noise_ratio_(imaging) 
        simple_snr_array = cropped_video.mean(axis=(1,2)) / cropped_video.std(axis=(1,2))
        
        return {'simple_snr_mean': simple_snr_array.mean(),
                'simple_snr_med': np.median(simple_snr_array),
                'simple_snr_std': simple_snr_array.std()}

    def get_photon_metrics(self, start_frame=1, end_frame=2001, crop=(30,30)):
        """Photon Metrics.
        
        Compute metrics related to the physio signal. Helper function for _calculate_qc_metrics
        
        Args:  
            start_frame, end_frame: Integers specifying the beginning and end frame to use
            crop:  A tuple (px_y, px_x) specifying the number of edge pixels to remove

        Returns:  
            A dictionary of metrics
        """
        cropped_video = self.subsample_and_crop_video(subsample=1,
                                                      start_frame=start_frame,
                                                      end_frame=end_frame,
                                                      crop=self.validate_crop(crop))
        
        photon_gain_dict = self.get_photon_gain_parameters(start_frame=start_frame, end_frame=end_frame)
        background_noise_mean = -photon_gain_dict['offset'] / photon_gain_dict['slope']
        photon_flux = (cropped_video.flatten() - background_noise_mean) / photon_gain_dict['slope']

        return {'photon_flux_median': np.median(photon_flux),
                'photon_gain': photon_gain_dict['slope'],
                'background_noise': background_noise_mean,
                'photon_offset': photon_gain_dict['offset']}

    def get_photon_gain_parameters(self, start_frame=1, end_frame=2001, crop=(150,150), perc_min=3, perc_max=90):
        """Photon Gain.
        
        Compute a variety of parameters related to the physio signal's gain.
        
        Args: 
            start_frame, end_frame: Integers specifying the beginning and end frame to use
            crop:  A tuple (px_y, px_x) specifying the number of edge pixels to remove
            perc_min, perc_max:  Min and max values between 0-100 used in filtering based on percentile

        Returns: 
            A dictionary of parameters related to the physio signal.  Useful in making plots and metrics.
        """
        cropped_video = self.subsample_and_crop_video(subsample=1,
                                                      start_frame=start_frame,
                                                      end_frame=end_frame,
                                                      crop=self.validate_crop(crop))

        # Remove saturated pixels
        dynamic_range = self.get_dynamic_range()
        idxs_not_saturated = np.where(cropped_video.max(axis=0).flatten() < dynamic_range[1])

        _var = cropped_video.var(axis=0).flatten()[idxs_not_saturated]
        _mean = cropped_video.mean(axis=0).flatten()[idxs_not_saturated]

        # Remove pixels that deviate from Poisson stats
        _var_scale = np.percentile(_var, [perc_min, perc_max])
        _mean_scale = np.percentile(_mean, [perc_min, perc_max])
        
        # Remove outliers
        _var_bool = np.logical_and(_var > _var_scale[0], _var < _var_scale[1])
        _mean_bool = np.logical_and(_mean > _mean_scale[0], _mean < _mean_scale[1])
        _no_outliers = np.logical_and(_var_bool, _mean_bool)

        _var_filt = _var[_no_outliers]
        _mean_filt = _mean[_no_outliers]
        _mat = np.vstack([_mean_filt, np.ones(len(_mean_filt))]).T
        try:
            slope, offset = np.linalg.lstsq(_mat, _var_filt, rcond=None)[0]
        except LinAlgError:
            raise DataCorruptionError('Unable to get photon metrics - check video for anomalies.')

        return {'var': _var_filt,
                'mean': _mean_filt,
                'slope': slope,
                'offset': offset}