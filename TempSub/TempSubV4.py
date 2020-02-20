#!/usr/bin/env python
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx #
# xxxxxxxxxxxxxxxxxxxxxx--------------------------TEMPLATE SUBTRACTION------------------------xxxxxxxxxxxxxxxxxxxxxxx #
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx #


# ------------------------------------------------------------------------------------------------------------------- #
# Import Required Libraries
# ------------------------------------------------------------------------------------------------------------------- #
import os
import re
import glob
import shutil
import warnings
import subprocess
import numpy as np
from pyraf import iraf

import astropy.wcs as wcs
from astropy.wcs import WCS
from astropy import units as u
from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
from astropy.convolution import Gaussian2DKernel
from photutils import Background2D, MedianBackground
from astropy.stats import SigmaClip, sigma_clipped_stats

from scipy import ndimage
import matplotlib.pyplot as plt
from image_registration import chi2_shift
from image_registration.fft_tools import shift

warnings.filterwarnings('ignore')
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
# Define The Global Variables Here
# ------------------------------------------------------------------------------------------------------------------- #
OBJECT_keyword = 'OBJECT'
RA_keyword = 'TARRA'
DEC_keyword = 'TARDEC'

platescale_sci = 0.6765
# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
# Functions For Handling Files & Lists
# ------------------------------------------------------------------------------------------------------------------- #


def remove_file(file_name):
    """
    Removes the file 'file_name' in the constituent directory.
    Args:
         file_name  : Name of the file to be removed from the current directory
    Returns:
        None
    """
    try:
        os.remove(file_name)
    except OSError:
        pass


def remove_similar_files(common_text):
    """
    Removes similar files based on the string 'common_text'.
    Args:
        common_text : String containing partial name of the files to be deleted
    Returns:
        None
    """
    for residual_file in glob.glob(common_text):
        remove_file(residual_file)


def group_similar_files(text_list, common_text, exceptions=''):
    """
    Groups similar files based on the string 'common_text'. Writes the similar files
    onto the list 'text_list' (only if this string is not empty) and appends the similar
    files to a list 'python_list'.
    Args:
        text_list   : Name of the output text file with names grouped based on the 'common_text'
        common_text : String containing partial name of the files to be grouped
        exceptions  : String containing the partial name of the files that need to be excluded
    Returns:
        list_files  : Python list containing the names of the grouped files
    """
    list_files = glob.glob(common_text)
    if exceptions != '':
        list_exception = exceptions.split(',')
        for file_name in glob.glob(common_text):
            for text in list_exception:
                test = re.search(text, file_name)
                if test:
                    try:
                        list_files.remove(file_name)
                    except ValueError:
                        pass

    list_files.sort()
    if len(text_list) != 0:
        with open(text_list, 'w') as f:
            for file_name in list_files:
                f.write(file_name + '\n')

    return list_files

# ------------------------------------------------------------------------------------------------------------------- #

# ------------------------------------------------------------------------------------------------------------------- #
# Define The Science & Reference Images
# Obtain Data and Header Details
# ------------------------------------------------------------------------------------------------------------------- #

# ref_name = 'ps1-g.fits'
# ref_name = 'ps1-r.fits'
# ref_name = 'ps1-i.fits'

# sci_name = '20190425215436-506-g1.fits'
# sci_name = '20190425220134-163-g2.fits'
# sci_name = '20190425225442-697-g3.fits'
# sci_name = '20190425230641-557-g4.fits'
# sci_name = '20190425180639-863-r1.fits'
# sci_name = '20190425182539-237-r2.fits'
# sci_name = '20190425220850-548-r3.fits'
# sci_name = '20190425224029-388-i1.fits'


def run_tempsub(sci_name, ref_name):

    image_ref = os.path.join(DIR_DATA, ref_name)
    image_sci = os.path.join(DIR_DATA, sci_name)
    print ("\n" + "# " + "-" * 40 + " #")
    print ("Science Image : {0}".format(image_sci))
    print ("Reference Image : {0}".format(image_ref))
    print ("# " + "-" * 40 + " #\n")

    hdu_sci = fits.open(image_sci)
    hdu_ref = fits.open(image_ref)
    hdr_sci = hdu_sci[0].header
    hdr_ref = hdu_ref[0].header
    data_ref = hdu_ref[0].data
    # --------------------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------------------- #
    # Print The Object Details
    # Print The Details Of the Reference Frame
    # --------------------------------------------------------------------------------------- #
    NAME_obj = hdr_sci[OBJECT_keyword]
    RA_obj = hdr_sci[RA_keyword]
    DEC_obj = hdr_sci[DEC_keyword]

    print ("\n" + "# " + "-" * 40 + " #")
    print ("Target Name : {0}".format(NAME_obj))
    print ("Target RA : {0}".format(RA_obj))
    print ("Target DEC : {0}ss".format(DEC_obj))
    print ("# " + "-" * 40 + " #\n")

    wcs_ref = WCS(hdr_ref)
    [RA_ref, DEC_ref] = wcs_ref.all_pix2world(data_ref.shape[0] / 2, data_ref.shape[1] / 2, 1)

    print ("\n" + "# " + "-" * 40 + " #")
    print ("Reference Frame Details : ")
    print ("Target RA : {0}".format(RA_ref))
    print ("Target DEC : {0}".format(DEC_ref))
    print ("# " + "-" * 40 + " #\n")
    # --------------------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------------------- #
    # Run Swarp To Astrometrically Align The Images
    # --------------------------------------------------------------------------------------- #
    exec_swarp = """swarp {0:s} {1:s} -c {2:s} -SUBTRACT_BACK N -RESAMPLE Y -RESAMPLE_DIR .
    -COMBINE N -IMAGE_SIZE 800,800 -CENTER_TYPE MANUAL -CENTER {3:0.3f},
    {4:0.3f}""".format(image_sci, image_ref, os.path.join(DIR_DATA, 'config.swarp'), RA_ref, DEC_ref)

    print ("\n" + "# " + "-" * 60 + " #")
    print ("Running Swarp...")
    print ("Executing command : {0}".format(exec_swarp))
    os.system(exec_swarp)
    print ("# " + "-" * 60 + " #\n")

    resamp_sci = image_sci.replace('.fits', '.resamp.fits').replace('data', 'processed')
    resamp_ref = image_ref.replace('.fits', '.resamp.fits').replace('data', 'processed')
    # --------------------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------------------- #
    # Read Data & Header Of Resampled Images
    # --------------------------------------------------------------------------------------- #
    hdu_resamp_ref = fits.open(resamp_ref)
    hdu_resamp_sci = fits.open(resamp_sci)

    header_ref = hdu_resamp_ref[0].header
    data_ref = hdu_resamp_ref[0].data

    print ("\n" + "# " + "-" * 40 + " #")
    print ("Reference Image Dimensions :")
    print ("X-Size : {0}".format(header_ref['NAXIS1']))
    print ("Y-Size : {0}".format(header_ref['NAXIS2']))
    print ("# " + "-" * 40 + " #\n")

    header_sci = hdu_resamp_sci[0].header
    data_sci = hdu_resamp_sci[0].data

    print ("\n" + "# " + "-" * 40 + " #")
    print ("Science Image Dimensions :")
    print ("X-Size : {0}".format(header_sci['NAXIS1']))
    print ("Y-Size : {0}".format(header_sci['NAXIS2']))
    print ("# " + "-" * 40 + " #\n")

    for (keyword, value) in hdr_sci.items():
        if keyword in header_sci.keys():
            header_sci.remove(keyword, remove_all=True)
            header_sci.append(card=(keyword, value))
    hdu_resamp_sci.flush()
    hdu_resamp_sci.close()

    # image_sub = hdu_resamp_ref[0].data - hdu_resamp_sci[0].data
    # hdu_image_sub=fits.PrimaryHDU(image_sub)
    # hdu_image_sub.writeto('sub_test_0.fits')
    # --------------------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------------------- #
    # Subtracting the Background From Science & Reference Frame
    # --------------------------------------------------------------------------------------- #
    bg_sci = os.path.join(DIR_PROC, 'bg_' + resamp_sci.split('/')[-1])
    bg_ref = os.path.join(DIR_PROC, 'bg_' + resamp_ref.split('/')[-1])

    if os.path.exists(bg_sci):
        os.remove(bg_sci)
    if os.path.exists(bg_ref):
        os.remove(bg_ref)

    mean_sci, median_sci, std_sci = sigma_clipped_stats(data_sci, sigma=3)
    hdu_bg_sci = fits.PrimaryHDU(data_sci - median_sci, header_sci)
    hdu_bg_sci.writeto(bg_sci)
    sci_data = hdu_bg_sci.data

    mean_ref, median_ref, std_ref = sigma_clipped_stats(data_ref, sigma=3)
    hdu_bg_ref = fits.PrimaryHDU(data_ref - median_ref, header_ref)
    hdu_bg_ref.writeto(bg_ref)

    print ("\n" + "# " + "-" * 40 + " #")
    print ("Background Count Details :")
    print ("Science Frame Median : {0}".format(median_sci))
    print ("Reference Frame Median : {0}".format(median_ref))
    print ("# " + "-" * 40 + " #\n")
    # --------------------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------------------- #
    # Perform PSF matching
    # --------------------------------------------------------------------------------------- #
    if os.path.exists('sci_prepsfex.cat'):
        os.remove('sci_prepsfex.cat')

    exec_sex = """sex {0:s} -c {1:s} -CATALOG_NAME {2:s} -MAG_ZEROPOINT 25.0 -SATUR_LEVEL 50000 -GAIN 1.6
    -CATALOG_TYPE=ASCII_HEAD""".format(bg_sci, os.path.join(DIR_DATA, 'prepsfex.sex'),
                                       os.path.join(DIR_DATA, 'sci_prepsfex.cat'))

    print ("\n" + "# " + "-" * 60 + " #")
    print ("Running Sextractor On The Science Image ...")
    print ("Executing command : {0}".format(exec_sex))
    os.system(exec_sex)
    print ("# " + "-" * 60 + " #\n")
    shutil.copy2(os.path.join(DIR_DATA, 'sci_prepsfex.cat'), os.path.join(DIR_PROC, 'sci_prepsfex.cat'))

    if os.path.exists('ref_prepsfex.cat'):
        os.remove('ref_prepsfex.cat')

    exec_sex2 = """sex {0:s} -c {1:s} -CATALOG_NAME {2:s} -MAG_ZEROPOINT 25.0 -SATUR_LEVEL 500000 -GAIN 1.01
    -CATALOG_TYPE=ASCII_HEAD""".format(bg_ref, os.path.join(DIR_DATA, 'prepsfex.sex'),
                                       os.path.join(DIR_DATA, 'ref_prepsfex.cat'))

    print ("\n" + "# " + "-" * 60 + " #")
    print ("Running Sextractor On The Reference Image ...")
    print ("Executing command : {0}".format(exec_sex2))
    os.system(exec_sex2)
    print ("# " + "-" * 60 + " #\n")
    shutil.copy2(os.path.join(DIR_DATA, 'ref_prepsfex.cat'), os.path.join(DIR_PROC, 'ref_prepsfex.cat'))

    catalog_sci = ascii.read(os.path.join(DIR_PROC, 'sci_prepsfex.cat'))
    clean_sources_sci = catalog_sci[(catalog_sci['FLAGS'] == 0) & (catalog_sci['SNR_WIN'] > 10)]
    fwhm_sci = np.median(clean_sources_sci['FWHM_WORLD'])

    catalog_ref = ascii.read(os.path.join(DIR_PROC, 'ref_prepsfex.cat'))
    clean_sources_ref = catalog_ref[(catalog_ref['FLAGS'] == 0) & (catalog_ref['SNR_WIN'] > 10)]
    fwhm_ref = np.median(clean_sources_ref['FWHM_WORLD'])

    fwhm_diff = 3600 * np.sqrt(fwhm_sci ** 2 - fwhm_ref ** 2)
    sigma = fwhm_diff / (platescale_sci * 2.3546)

    print (clean_sources_sci['FWHM_WORLD'] * 3600 / platescale_sci)
    print (clean_sources_ref['FWHM_WORLD'] * 3600 / platescale_sci)

    print ("\n" + "# " + "-" * 60 + " #")
    print ("FWHM of the Science Frame : {0:.2f}".format(fwhm_sci * 3600 / platescale_sci))
    print ("FWHM of the Reference Frame : {0:.2f}".format(fwhm_ref * 3600 / platescale_sci))
    print ("Sigma of the Convolution Kernel : {0:.2f}".format(sigma))
    print ("# " + "-" * 60 + " #\n")

    if not os.path.isdir('out'):
        os.mkdir('out')
    # --------------------------------------------------------------------------------------- #

    def gauss(file_name, sigma, prefix_str='conv_'):
        """
        Convolve the input image with an elliptical gaussian function.
        Args:
            file_name       : Input image to be convolved
            sigma           : Sigma of gaussian along major axis of ellipse
            prefix_str      : Prefix to distinguish the aligned FITS file from the original FITS file
        Returns:
            output_filename : Name of the convolved FITS file
        """
        task = iraf.images.imfilter.gauss
        task.unlearn()

        output_filename = os.path.join(DIR_PROC, prefix_str + file_name.split('/')[-1])
        try:
            os.remove(output_filename)
        except OSError:
            pass
        task(input=file_name, output=output_filename, sigma=sigma)

        return output_filename

    conv_ref = gauss(bg_ref, sigma)
    data_convref = fits.open(conv_ref)[0].data
    print (conv_ref)
    # --------------------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------------------- #
    # Run Sextractor To Extract Sources For Alignment and Scaling
    # Precise Align The Images
    # --------------------------------------------------------------------------------------- #

    exec_sex3 = """sex -c {0:s} {1:s} {2:s} -CATALOG_NAME {3:s} -MAG_ZEROPOINT 25.0
    -CATALOG_TYPE=ASCII_HEAD""".format(os.path.join(DIR_PROC, 'science.sex'), os.path.join(DIR_DATA, 'science.param'),
                                       resamp_sci, os.path.join(DIR_PROC, 'sci_match.cat'))

    print ("\n" + "# " + "-" * 60 + " #")
    print ("Running Sextractor On The Science Image ...")
    print ("Executing command : {0}".format(exec_sex3))
    os.system(exec_sex3)
    print ("# " + "-" * 60 + " #\n")

    exec_sex4 = """sex {0:s} -c {1:s} -CATALOG_NAME {2:s} -MAG_ZEROPOINT 25.0
    -CATALOG_TYPE=ASCII_HEAD""".format(conv_ref, os.path.join(DIR_PROC, 'reference.sex'),
                                       os.path.join(DIR_PROC, 'ref_match.cat'))
    print ("\n" + "# " + "-" * 60 + " #")
    print ("Running Sextractor On The Convolved Reference Image ...")
    print ("Executing command : {0}".format(exec_sex4))
    os.system(exec_sex4)
    print ("# " + "-" * 60 + " #\n")

    # Fine tuning image alignment
    # xoff, yoff, exoff, eyoff = chi2_shift(hdu_bg_ref.data, hdu_bg_sci.data, 10,
    #                                      return_error=True, upsample_factor='auto')
    # sci_shift = ndimage.shift(hdu_bg_sci.data, [-yoff, -xoff], order=3, mode='reflect', cval=0.0, prefilter=True)
    # --------------------------------------------------------------------------------------- #

    # --------------------------------------------------------------------------------------- #
    # Perform Flux Scaling
    # --------------------------------------------------------------------------------------- #
    cat_sci = ascii.read('sci_match.cat')
    cat_ref = ascii.read('ref_match.cat')
    if os.path.exists('prepsfex.cat'):
        os.remove('prepsfex.cat')

    source_sci = SkyCoord(ra=cat_sci['X_WORLD'], dec=cat_sci['Y_WORLD'], unit='degree')
    source_ref = SkyCoord(ra=cat_ref['X_WORLD'], dec=cat_ref['Y_WORLD'], unit='degree')

    idx, d2d, d3d = source_sci.match_to_catalog_sky(source_ref)

    index_arr = []
    ratio_arr = []
    for idx1, idx2, d in zip(idx, np.arange(len(d2d)), d2d):
        index_arr.append(idx1)
        # print (cat_ref['X_WORLD'][i],cat_ref['Y_WORLD'][i],'  ', cat_sci['X_WORLD'][i2],cat_sci['Y_WORLD'][i2])
        # print (cat_ref['FLUX_AUTO'][i], cat_sci['FLUX_AUTO'][i2] )
        ratio_arr.append(cat_sci['FLUX_AUTO'][idx2] / cat_ref['FLUX_AUTO'][idx1])

    scale = np.median(ratio_arr)
    print ("Flux Scaling Factor Is : {0}".format(scale))

    final_ref = fits.PrimaryHDU(scale * data_convref, header_ref)
    remove_file(os.path.join(DIR_PROC, 'final_' + conv_ref.split('/')[-1]))
    final_ref.writeto(os.path.join(DIR_PROC, 'final_' + conv_ref.split('/')[-1]))

    # image_sub = sci_shift - scale * data_convref
    image_sub = hdu_bg_sci.data - scale * data_convref
    hdu_image_sub = fits.PrimaryHDU(image_sub, hdr_sci)
    remove_file(os.path.join(DIR_PROC, 'sub_' + bg_sci.split('/')[-1]))
    hdu_image_sub.writeto(os.path.join(DIR_PROC, 'sub_' + bg_sci.split('/')[-1]))

    # --------------------------------------------------------------------------------------- #


ref_names = {'g': 'ps1-g.fits', 'r': 'ps1-r.fits', 'i': 'ps1-i.fits'}
sci_names = {'g': ['20190425215436-506-g1.fits', '20190425220134-163-g2.fits',
             '20190425225442-697-g3.fits', '20190425230641-557-g4.fits'],
             'r': ['20190425180639-863-r1.fits', '20190425182539-237-r2.fits',
                   '20190425220850-548-r3.fits'], 'i': ['20190425224029-388-i1.fits']}


# ------------------------------------------------------------------------------------------------------------------- #
# Set The Directory Structure & Remove Residual Files From Previous Run
# ------------------------------------------------------------------------------------------------------------------- #
DIR_CURNT = os.getcwd()
DIR_PROC = os.path.join(DIR_CURNT, 'processed')
DIR_DATA = os.path.join(DIR_CURNT, 'data')
DIR_OUT = os.path.join(DIR_PROC, 'out')

# [os.remove(filename) for filename in os.listdir(DIR_CURNT) if filename.endswith('.fits')]

# if os.path.isdir(DIR_PROC):
#     shutil.rmtree(DIR_PROC)
# os.mkdir(DIR_PROC)

for filename in os.listdir(DIR_DATA):
    shutil.copy2(os.path.join(DIR_DATA, filename), os.path.join(DIR_PROC, filename))
os.chdir(DIR_PROC)

remove_similar_files(common_text='*.resamp.fits')

# ------------------------------------------------------------------------------------------------------------------- #

for band, ref_name in ref_names.items():
    print (band, ref_name)
    for sci_name in sci_names[band]:
        print (sci_name, ref_name)
        run_tempsub(sci_name, ref_name)

# ------------------------------------------------------------------------------------------------------------------- #
