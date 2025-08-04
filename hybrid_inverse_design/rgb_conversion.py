import numpy as np
import colour
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def load_cie_data(target_wvl):
    """Load and interpolate CIE 1931 colour matching functions (MSDS_CMFS) and D65 ilumination (white light, SDS_ILLUMINANTS)
        Args: 
            target_wvl: wavelength range of the reflection spectrum [nm]
    """
    cmfs = colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"] # x is red/green response, y is luminance sensitivity and z is blue response
    d65 = colour.SDS_ILLUMINANTS["D65"] # Needed to scale the luminance of the colour
    
    cie_wavelengths = cmfs.wavelengths
    x_bar, y_bar, z_bar = cmfs.values.T
    d65_wavelengths = d65.wavelengths
    d65_values = d65.values

    # Interpolate to match the target wavelength range and step
    interp_x_bar = interp1d(cie_wavelengths, x_bar, kind="linear")
    interp_y_bar = interp1d(cie_wavelengths, y_bar, kind="linear")
    interp_z_bar = interp1d(cie_wavelengths, z_bar, kind="linear")
    interp_d65 = interp1d(d65_wavelengths, d65_values, kind="linear")

    return (target_wvl,
            interp_x_bar(target_wvl),
            interp_y_bar(target_wvl),
            interp_z_bar(target_wvl),
            interp_d65(target_wvl))

def spectrum_to_rgb(target_wvl, reflectance):
    """Convert reflectance spectrum to sRGB (in [0, 1] and [0, 255] ranges)
        Args:
            target_wvl: wavelength range of the reflection spectrum [nm];
            reflectance: R values.
        """
    cie_wvl, x_bar, y_bar, z_bar, d65 = load_cie_data(target_wvl)

    X = np.trapz(reflectance * x_bar * d65, cie_wvl)
    Y = np.trapz(reflectance * y_bar * d65, cie_wvl)
    Z = np.trapz(reflectance * z_bar * d65, cie_wvl)
    #___________________________LUMINANCE_______________________________
    Y_ref = np.trapz(y_bar * d65, cie_wvl) # Y_ref represents maximum brightness
    # XYZ normalization by the total luminance of the reference white
    X /= Y_ref
    Y /= Y_ref
    Z /= Y_ref

    # sRGB matrix
    xyz_to_rgb_matrix = np.array([
        [ 3.2406, -1.5372, -0.4986],
        [-0.9689,  1.8758,  0.0415],
        [ 0.0557, -0.2040,  1.0570]
    ])
    rgb_linear = np.dot(xyz_to_rgb_matrix, [X, Y, Z]) # if error, use np.matmul (preferred, but was not working???)

    # Human vision brightness correction (more sensitive to darker colors than bright)
    def gamma_correct(c):
        return 12.92 * c if c <= 0.0031308 else 1.055 * (c ** (1 / 2.4)) - 0.055

    rgb = np.clip([gamma_correct(c) for c in rgb_linear], 0, 1) # r, g, b between [0, 1]
    rgb_255 = tuple(int(round(c * 255)) for c in rgb) # r, g, b between [0, 255], integer

    return rgb, rgb_255