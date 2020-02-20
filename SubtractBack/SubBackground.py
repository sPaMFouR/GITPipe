#!/usr/bin/env python
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx #
# xxxxxxxxxxxxxxxxxx-------------------BACKGROUND SUBTRACTION OF OBJECT FRAMES---------------------xxxxxxxxxxxxxxxxxx #
# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx #


# ------------------------------------------------------------------------------------------------------------------- #
# Import Required Libraries
# ------------------------------------------------------------------------------------------------------------------- #
import os
import re
import glob
import warnings
from astropy.io import fits
from astropy.stats import sigma_clipped_stats

warnings.filterwarnings('ignore')
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
# Function To Estimate Background And Subtract
# ------------------------------------------------------------------------------------------------------------------- #

def estimate_background(filename):

    hdulist = fits.open(filename)
    header = hdulist[0].header
    data = hdulist[0].data

    mean, median, std = sigma_clipped_stats(data, sigma=3)
    # hdu_bgdata = fits.PrimaryHDU(data - median, header)
    # hdu_bgdata.writeto('bg_' + filename)

    exec_sex = """sex {0:s} -c {1:s} -BACK_SIZE 50 -BACK_FILTERSIZE 3 -CHECKIMAGE_TYPE -BACKGROUND
    -CHECKIMAGE_NAME {2:s}""".format(filename, 'config.sex', 'bgs_' + filename)

    print ("\n" + "# " + "-" * 60 + " #")
    print ("Background Count Details :")
    print ("Image Name : {0}".format(filename))
    print ("Image Median : {0:0.1f}".format(median))
    print ("Running Sextractor On The Image ...")
    print ("Executing command : {0}".format(exec_sex))
    os.system(exec_sex)
    print ("# " + "-" * 60 + " #\n")


# ------------------------------------------------------------------------------------------------------------------- #
# Run Background Subtraction
# ------------------------------------------------------------------------------------------------------------------- #
list_files = group_similar_files('', common_text='HD*.fits')

for file_name in list_files:
    estimate_background(file_name)
# ------------------------------------------------------------------------------------------------------------------- #
