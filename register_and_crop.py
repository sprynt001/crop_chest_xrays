"""Register and crop images
This script allows registration of images to a reference and cropping of
a ROI in the reference space.

It was written to allow the extraction of the chest region from whole
body x-rays of mice.

The key function is register_image which can be imported i.e.

from register_and_crop import register_image

However, the script can be called with parameters to allow registration of
a single image or a number of images whose paths are in a text file. Type

python register_and_crop.py --help for more info.
"""

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk

def register_image(fixed_image, moving_image):
    """register moving_image to fixed_image using rigid body registration.
    
    Parameters
    ----------
    fixed_image : SimpleITK.Image of type sitkFloat32
        Reference image.
    moving_image : SimpleITK.Image of type sitkFloat32
        Image to be registered.

    Returns
    -------
    moving_resampled : SimpleITK.Image of type sitkFloat32
        Moving image transformed to same space as reference image.

    final_transform : SimpleITK.Transform
        The transform object with the transform from the moving image to
        the reference image.
    
    """

    # Initially align the centers
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image, 
        moving_image, 
        sitk.Euler2DTransform())

    # registration configuration from multires_registration function. See
    # http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/60_Registration_Introduction.html
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(
        numberOfHistogramBins=50)
    registration_method.SetMetricSamplingStrategy(
        registration_method.RANDOM)
    # SimpleITK example was sampling at 0.01 - increased this to get 
    # more reliable registration
    registration_method.SetMetricSamplingPercentage(0.1)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=100,
        estimateLearningRate=registration_method.Once)
    registration_method.SetOptimizerScalesFromPhysicalShift() 
    registration_method.SetInitialTransform(
        initial_transform, inPlace=False)
    # Use 4 pyramid levels (instead of 3 as in the SimpleITK examples)
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [8,4,2,1])
    registration_method.SetSmoothingSigmasPerLevel(
        smoothingSigmas = [4,2,1,0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    final_transform = registration_method.Execute(fixed_image, moving_image)
    print("Final metric value: {0}".format(
        registration_method.GetMetricValue()))
    print("Optimizer's stopping condition, {0}".format(
        registration_method.GetOptimizerStopConditionDescription()))

    # Apply transform to get aligned version of moving image
    moving_resampled = sitk.Resample(moving_image, 
                                     fixed_image,
                                     final_transform,
                                     sitk.sitkLinear, 0.0,
                                     moving_image.GetPixelID())
    return (moving_resampled, final_transform)

# Utility function to make 2D images of size (nrows, ncols, 1) into
# (nrows, ncols) i.e. squeeze out the redundant dimension
def squeeze_image(im):
    """Squeeze third dimension of sitk images (size 1) so they are 2D images
    
    """
    im_numpy = sitk.GetArrayFromImage(im)
    im_sitk = sitk.GetImageFromArray(np.squeeze(im_numpy))
    return im_sitk

def plot_rect(left, top, width, height, color='r', ax=None):
    """Plot rectangle on given axis object
    
    """    
    right = left + width
    bottom = top + height
    x = [left, left,   right,  right, left]
    y = [top,  bottom, bottom, top,   top]
    if ax == None:
        ax = plt.gca()
    ax.plot(x, y, color)

def crop_image(im, left, top, width, height):
    """crop an image
    
    """    
    right = left + width
    bottom = top + height

    im_numpy = sitk.GetArrayFromImage(im)
    im_sitk = sitk.GetImageFromArray(im_numpy[top:bottom,left:right])
    return im_sitk

if __name__ == "__main__":
    import sys
    import os
    import argparse

    parser = argparse.ArgumentParser(
        "Register an image or a list of images to a fixed (reference) image"
    )
    
    parser.add_argument("-f", "--fixed-image", required=True,
                        dest="fixed_image", 
                        help="Path to fixed (Reference) image"
    )
    parser.add_argument("-o", "--output-dir", dest="output_dir",
        default="./",
        help="Path to the output directory to store the registered image"
    )
    parser.add_argument("-c", "--crop-params", dest="crop_params", 
        help="If an ROI is required parameters for this as " + \
             "left,top,width,height e.g. 890,400,700,700"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-m", "--moving-image", dest="moving_image", 
        help="Path to image to be registered. Note that this " + \
             "parameter should not be used with '--moving-images'!"
    )
    group.add_argument("--moving-images", dest="moving_images",
        help="Path to a text file whose lines contain paths to " + \
             "images to be registered. This parameter should not be " + \
             "used with '-m' or '--moving-image'!"
    )
    
    args = parser.parse_args()

    # Parse the crop_parameters if supplied
    if args.crop_params is not None:
        crop_flag = True
        try:
            crop_params = [int(c) for c in args.crop_params.split(",")]
            if len(crop_params) != 4:
                message = "Crop parameters should be comma separated " + \
                      "list of four integers: left,top,width,height"
                raise Exception(message)
            left = crop_params[0]
            top = crop_params[1]
            width = crop_params[2]
            height = crop_params[3]
        except Exception as e:
            print(f"Error with crop parameters: {e}")
            parser.print_usage()
            sys.exit(-1)
    else:
        crop_flag = False


    fixed_image = sitk.ReadImage(args.fixed_image, sitk.sitkFloat32)
    if fixed_image.GetDimension() == 3:
        fixed_image = squeeze_image(fixed_image)

    if args.moving_image is not None:
        moving_image_paths = [args.moving_image,]
    else:
        # Read in paths from file
        with open(args.moving_images, "rt") as fid:
            moving_image_paths = [p.strip('\n') for p in fid.readlines()]

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Output dir {args.output_dir} has been created")
    
    n_images_to_register = len(moving_image_paths)

    for i, moving_image_path in enumerate(moving_image_paths, start=1):
        print(f"Processing image {i} of {n_images_to_register}")
        
        # Images will be processed as float - but Dicom only saves in int
        # Read first image in original format and store the type so it
        # can be written out with same type.
        # Create cast filter to convert images back to original format
        if i == 1:
            moving_image = sitk.ReadImage(moving_image_path)
            castFilter = sitk.CastImageFilter()
            castFilter.SetOutputPixelType(moving_image.GetPixelID())

        moving_image = sitk.ReadImage(moving_image_path, sitk.sitkFloat32)
        if moving_image.GetDimension() == 3:
            moving_image = squeeze_image(fixed_image)
        moving_resampled,_ = register_image(fixed_image, moving_image)

        if crop_flag:
            moving_resampled = crop_image(moving_resampled, 
                                          left, top, width, height)
            fname = "cropped_registered_" + \
                os.path.split(moving_image_path)[-1]
        else:
            fname = "registered_" + os.path.split(moving_image_path)[-1]

        output_path = os.path.join(args.output_dir,fname)
        sitk.WriteImage(castFilter.Execute(moving_resampled), output_path)
        print(f"Saved registered image to {output_path}")
