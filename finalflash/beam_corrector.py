from astropy.io import fits
import numpy as np
import argparse
from datetime import datetime

# Define polynomial coefficients for each band (Updated with new coefficients)
POLY_COEFFS = {
    'Band-2': (-2.83, 33.564, -18.026, 3.588),  # Updated coefficients for Band 2
    'Band-3': (-2.939, 33.312, -16.659, 3.066),  # Updated coefficients for Band 3
    'Band-4': (-3.190, 38.642, -20.471, 3.964),  # Updated coefficients for Band 4
    'Band-5': (-2.608, 27.357, -13.091, 2.368),  # Updated coefficients for Band 5
}

def find_freq(header):
    """
    Find frequency value in the FITS header.
    """
    print("Searching for frequency in the header...")
    
    # Try to find frequency in common places
    for i in range(5):
        ctype_key = 'CTYPE%i' % i
        crval_key = 'CRVAL%i' % i
        if ctype_key in header and 'FREQ' in header[ctype_key]:
            freq = header.get(crval_key)
            return freq
    
    # If not found, look for specific keywords
    freq = header.get('RESTFRQ') or header.get('FREQ')
    if freq:
        return freq
    
    # If nothing found, print and return None
    print("Frequency not found in the header.")
    return None



def get_band_coefficients(frequency_ghz):
    """Determine the band's polynomial coefficients based on frequency in GHz."""
    if 0.125 <= frequency_ghz < 0.25:
        return POLY_COEFFS['Band-2']
    elif 0.25 <= frequency_ghz < 0.5:
        return POLY_COEFFS['Band-3']
    elif 0.55 <= frequency_ghz < 0.85:
        return POLY_COEFFS['Band-4']
    elif 1.05 <= frequency_ghz < 1.45:
        return POLY_COEFFS['Band-5']
    else:
        raise ValueError(f'Frequency {frequency_ghz} GHz not in known bands.')

def primary_beam_model(frequency_ghz, radius_arcmin, coeffs):
    """Calculate the primary beam model based on polynomial coefficients."""
    a, b, c, d = coeffs
    # Compute correction using the polynomial formula
    correction = 1 + (a / 1e3) * (radius_arcmin * frequency_ghz)**2 + \
                     (b / 1e7) * (radius_arcmin * frequency_ghz)**4 + \
                     (c / 1e10) * (radius_arcmin * frequency_ghz)**6 + \
                     (d / 1e13) * (radius_arcmin * frequency_ghz)**8
    return correction


# ANSI escape codes for colors
RESET = "\033[0m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"


def get_frequencies(header):
    """
    Extract all frequency information from the FITS header.
    Returns list of frequencies in Hz.
    """
    print("Searching for frequency in the header...")
    
    # Debug: Print all header keys containing 'FREQ'
    print("\nDebug: Available frequency keys:")
    for key in header.keys():
        if 'FREQ' in key:
            print(f"{key}: {header[key]}")
            
    # Check for numbered FREQ keys (FREQ0001, etc)
    freq_keys = sorted([key for key in header.keys() if key.startswith('FREQ') and len(key) > 4 
                       and not key.startswith('FREL') and not key.startswith('FREH')])
            
    if freq_keys:
        # Debug: Print found keys
        print("\nDebug: Using these FREQ keys:", freq_keys)
        frequencies = [float(header[key]) for key in freq_keys]
        print(f"Found {len(frequencies)} frequencies from numbered FREQ keys")
        return frequencies
    
    # If no FREQ keys, try spectral axis info
    if 'CRVAL3' in header and 'CDELT3' in header and 'NAXIS3' in header:
        print("\nDebug: No FREQ keys found, using spectral axis info")
        crval = float(header['CRVAL3'])  # Reference frequency
        cdelt = float(header['CDELT3'])  # Frequency increment
        naxis = int(header['NAXIS3'])    # Number of channels
        
        # Generate frequency array for cube
        frequencies = [crval + (j * cdelt) for j in range(naxis)]
        print(f"Found {len(frequencies)} frequencies from frequency axis")
        return frequencies
    
    # Finally try single frequency value
    freq = header.get('RESTFRQ') or header.get('FREQ') or header.get('CRVAL3')
    if freq:
        print("\nDebug: Using single frequency value")
        print("Found single frequency value")
        return [float(freq)]
    
    print("Frequency not found in the header.")
    return None



def flatten_cube(filename):
    """Flatten a FITS file but preserve frequency planes. Returns clean header and cube data."""
    from astropy.wcs import WCS

    with fits.open(filename) as f:
        # Debug prints
        print("DEBUG: Original header keys:")
        for key in f[0].header.keys():
            if 'FREQ' in key:
                print(f"{key}: {f[0].header[key]}")
                
        naxis = f[0].header['NAXIS']
        if naxis < 2:
            raise ValueError('Cannot make map from this FITS file.')

        w = WCS(f[0].header)
        wn = WCS(naxis=3)  # Keep 3 dimensions for cube

        # Copy spatial dimensions WCS info
        wn.wcs.crpix[0] = w.wcs.crpix[0]
        wn.wcs.crpix[1] = w.wcs.crpix[1]
        wn.wcs.cdelt[:2] = w.wcs.cdelt[0:2]
        wn.wcs.crval[:2] = w.wcs.crval[0:2]
        wn.wcs.ctype[0] = w.wcs.ctype[0]
        wn.wcs.ctype[1] = w.wcs.ctype[1]

        if naxis > 2:  # Handle frequency axis for cubes
            wn.wcs.crpix[2] = w.wcs.crpix[2]
            wn.wcs.cdelt[2] = w.wcs.cdelt[2]
            wn.wcs.crval[2] = w.wcs.crval[2]
            wn.wcs.ctype[2] = w.wcs.ctype[2]

        header = wn.to_header()
        header["NAXIS"] = 3 if naxis > 2 else 2
        header["NAXIS1"] = f[0].header['NAXIS1']
        header["NAXIS2"] = f[0].header['NAXIS2']
        if naxis > 2:
            header["NAXIS3"] = f[0].header['NAXIS3']

        # Copy over all FREQ keys
        for key in f[0].header.keys():
            if key.startswith('FREQ'):
                header[key] = f[0].header[key]

        # Copy all frequency-related keys
        for key in f[0].header.keys():
            if 'FREQ' in key and key not in header:
                print(f"Debug: Copying {key}")
                header[key] = f[0].header[key]

        # Debug prints
        print("\nDEBUG: New header keys:")
        for key in header.keys():
            if 'FREQ' in key:
                print(f"{key}: {header[key]}")

        # For cube, keep all frequency planes
        if naxis > 2:
            return header, f[0].data
        else:
            # For 2D, keep original flatten behavior
            return header, f[0].data[tuple([0] * (naxis-2) + [slice(None)]*2)]

def correct_fits_with_primary_beam(input_fits, output_fits, beam_threshold=0.01):
    """Apply primary beam correction to a FITS file."""
    print(f"{MAGENTA}Developed by Arpan Pal at NCRA-TIFR in 2024.{RESET}")
    print(f"{CYAN}Starting the attack !!!!{RESET}")
    print(f"{YELLOW}Gathering relevant information from the FITS image......................{RESET}")
    print(f"{CYAN}Starting uGMRT beam correction!{RESET}")


    # Get data and clean header using flatten_cube
    header, data = flatten_cube(input_fits)
    
    # Handle frequency detection based on data dimensionality
    if data.ndim == 2:
        # For 2D images, use the original find_freq
        frequency_hz = find_freq(header)
        if frequency_hz is None:
            raise ValueError('Frequency information (FREQ) not found in FITS header')
        frequencies_ghz = [frequency_hz/1e9]  # Convert to GHz
        print(f"{GREEN}Found frequency: {frequencies_ghz[0]:.2f} GHz{RESET}")
    else:
        # For cubes, use get_frequencies
        frequencies = get_frequencies(header)
        if not frequencies:
            raise ValueError('No frequency information found in FITS header')
        frequencies_ghz = [f/1e9 for f in frequencies]
        print(f"{GREEN}Found {len(frequencies_ghz)} frequency planes{RESET}")

    # Calculate pixel scale (in arcmin)
    pixel_scale_deg = abs(header['CDELT1'])  # Degrees per pixel
    pixel_scale_arcmin = pixel_scale_deg * 60  # Arcminutes per pixel
    print(f"{GREEN}Pixel scale: {pixel_scale_arcmin:.4f} arcmin/pixel{RESET}")
    
    # Generate radius grid (in arcmin)
    y, x = np.indices(data.shape[-2:])  # Use last two dimensions for spatial coords
    x_center = header['CRPIX1'] - 1
    y_center = header['CRPIX2'] - 1
    r = np.sqrt((x - x_center)**2 + (y - y_center)**2) * pixel_scale_arcmin

    # Create output array with same shape as input
    corrected_data = np.zeros_like(data)

    # Handle both 2D and cube cases
    if data.ndim == 2:
        # Single image case
        freq_ghz = frequencies_ghz[0]
        print(f"{GREEN}Processing single image at {freq_ghz:.3f} GHz{RESET}")
        
        coeffs = get_band_coefficients(freq_ghz)
        print(f"{GREEN}Using polynomial coefficients: {coeffs}{RESET}")
        
        beam = primary_beam_model(freq_ghz, r, coeffs)
        print(f"{GREEN}Calculated Beams!{RESET}")

        with np.errstate(divide='ignore', invalid='ignore'):
            corrected_data = np.where(np.abs(beam) >= beam_threshold, data / beam, 0)
        print(f"{GREEN}Beams applied to input FITS image !!!{RESET}")

    else:
        # Cube case
        num_planes = len(frequencies_ghz)
        print(f"{GREEN}Processing cube with {num_planes} frequency planes{RESET}")

        for i in range(num_planes):
            freq_ghz = frequencies_ghz[i]
            print(f"{CYAN}Processing plane {i+1}/{num_planes} at {freq_ghz:.3f} GHz{RESET}")
            
            try:
                coeffs = get_band_coefficients(freq_ghz)
                print(f"{GREEN}Using polynomial coefficients: {coeffs}{RESET}")
                
                beam = primary_beam_model(freq_ghz, r, coeffs)
                
                if data.ndim == 3:
                    plane = data[i, :, :]
                else:  # 4D case
                    plane = data[0, i, :, :]
                    
                with np.errstate(divide='ignore', invalid='ignore'):
                    corrected_plane = np.where(np.abs(beam) >= beam_threshold,
                                             plane / beam, 0)
                
                if data.ndim == 3:
                    corrected_data[i, :, :] = corrected_plane
                else:  # 4D case
                    corrected_data[0, i, :, :] = corrected_plane
                    
                print(f"{GREEN}Beam applied to plane {i+1}!{RESET}")
                
            except ValueError as e:
                print(f"{RED}Warning: Skipping frequency {freq_ghz} GHz - {str(e)}{RESET}")
                # Copy original data for skipped planes
                if data.ndim == 3:
                    corrected_data[i, :, :] = data[i, :, :]
                else:  # 4D case
                    corrected_data[0, i, :, :] = data[0, i, :, :]
    

    # Read original header to preserve all metadata
    with fits.open(input_fits) as f:
        original_header = f[0].header.copy()
    # Update header history
    original_header['HISTORY'] = f"finalflash v0.3.2 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # Write corrected data
    fits.writeto(output_fits, corrected_data, original_header, overwrite=True, output_verify='ignore')
    print(f"{GREEN}Primary beam corrected FITS {'cube' if data.ndim > 2 else 'image'} saved to {output_fits}{RESET}")
    print(f"{MAGENTA}Finalflash done 💥💥💥💥💥💥{RESET}")

def get_jvla_beam_coefficients(frequency_ghz):
    """
    Determine the JVLA beam polynomial coefficients based on frequency in GHz.
    Based on EVLA Memo 195 by Rick Perley (June 8, 2016).
    Returns a tuple of coefficients (A0, A2, A4, A6) for the beam polynomial.
    """
    # P-band (224-480 MHz)
    if 0.224 <= frequency_ghz < 0.480:
        # Select coefficients based on specific frequency
        if frequency_ghz < 0.246:
            return (1.0, -1.137e-3, 5.19e-7, -1.04e-10)  # 232 MHz
        elif frequency_ghz < 0.281:
            return (1.0, -1.130e-3, 5.04e-7, -1.02e-10)  # 246 MHz
        elif frequency_ghz < 0.296:
            return (1.0, -1.106e-3, 5.11e-7, -1.10e-10)  # 281 MHz
        elif frequency_ghz < 0.312:
            return (1.0, -1.125e-3, 5.27e-7, -1.14e-10)  # 296 MHz
        elif frequency_ghz < 0.328:
            return (1.0, -1.030e-3, 4.44e-7, -0.89e-10)  # 312 MHz
        elif frequency_ghz < 0.344:
            return (1.0, -0.980e-3, 4.25e-7, -0.87e-10)  # 328 MHz
        elif frequency_ghz < 0.357:
            return (1.0, -0.974e-3, 4.09e-7, -0.76e-10)  # 344 MHz
        elif frequency_ghz < 0.382:
            return (1.0, -0.996e-3, 4.23e-7, -0.79e-10)  # 357 MHz
        elif frequency_ghz < 0.392:
            return (1.0, -1.002e-3, 4.39e-7, -0.88e-10)  # 382 MHz
        elif frequency_ghz < 0.403:
            return (1.0, -1.067e-3, 5.13e-7, -1.12e-10)  # 392 MHz
        elif frequency_ghz < 0.421:
            return (1.0, -1.057e-3, 4.90e-7, -1.06e-10)  # 403 MHz
        elif frequency_ghz < 0.458:
            return (1.0, -1.154e-3, 5.85e-7, -1.33e-10)  # 421 MHz
        elif frequency_ghz < 0.470:
            return (1.0, -0.993e-3, 4.67e-7, -1.04e-10)  # 458 MHz
        else:
            return (1.0, -1.010e-3, 4.85e-7, -1.07e-10)  # 470 MHz
    
    # L-band (1-2 GHz)
    elif 1.0 <= frequency_ghz < 2.0:
        # Select coefficients based on specific frequency
        if frequency_ghz < 1.104:
            return (1.0, -1.529e-3, 8.69e-7, -1.88e-10)  # 1040 MHz
        elif frequency_ghz < 1.168:
            return (1.0, -1.486e-3, 8.15e-7, -1.68e-10)  # 1104 MHz
        elif frequency_ghz < 1.232:
            return (1.0, -1.439e-3, 7.53e-7, -1.45e-10)  # 1168 MHz
        elif frequency_ghz < 1.296:
            return (1.0, -1.450e-3, 7.87e-7, -1.63e-10)  # 1232 MHz
        elif frequency_ghz < 1.360:
            return (1.0, -1.428e-3, 7.62e-7, -1.54e-10)  # 1296 MHz
        elif frequency_ghz < 1.424:
            return (1.0, -1.449e-3, 8.02e-7, -1.74e-10)  # 1360 MHz
        elif frequency_ghz < 1.488:
            return (1.0, -1.462e-3, 8.23e-7, -1.83e-10)  # 1424 MHz
        elif frequency_ghz < 1.552:
            return (1.0, -1.455e-3, 7.92e-7, -1.63e-10)  # 1488 MHz
        elif frequency_ghz < 1.680:
            return (1.0, -1.435e-3, 7.54e-7, -1.49e-10)  # 1552 MHz
        elif frequency_ghz < 1.744:
            return (1.0, -1.443e-3, 7.74e-7, -1.57e-10)  # 1680 MHz
        elif frequency_ghz < 1.808:
            return (1.0, -1.462e-3, 8.02e-7, -1.69e-10)  # 1744 MHz
        elif frequency_ghz < 1.872:
            return (1.0, -1.488e-3, 8.38e-7, -1.83e-10)  # 1808 MHz
        elif frequency_ghz < 1.936:
            return (1.0, -1.486e-3, 8.26e-7, -1.75e-10)  # 1872 MHz
        else:
            return (1.0, -1.459e-3, 7.93e-7, -1.62e-10)  # 1936 MHz
    
    # S-band (2-4 GHz)
    elif 2.0 <= frequency_ghz < 4.0:
        # Select coefficients based on specific frequency
        if frequency_ghz < 2.180:
            return (1.0, -1.429e-3, 7.52e-7, -1.47e-10)  # 2052 MHz
        elif frequency_ghz < 2.308:
            return (1.0, -1.389e-3, 7.06e-7, -1.33e-10)  # 2180 MHz
        elif frequency_ghz < 2.436:
            return (1.0, -1.377e-3, 6.90e-7, -1.27e-10)  # 2436 MHz
        elif frequency_ghz < 2.564:
            return (1.0, -1.377e-3, 6.90e-7, -1.27e-10)  # 2436 MHz
        elif frequency_ghz < 2.692:
            return (1.0, -1.381e-3, 6.92e-7, -1.26e-10)  # 2564 MHz
        elif frequency_ghz < 2.820:
            return (1.0, -1.402e-3, 7.23e-7, -1.40e-10)  # 2692 MHz
        elif frequency_ghz < 2.948:
            return (1.0, -1.433e-3, 7.62e-7, -1.54e-10)  # 2820 MHz
        elif frequency_ghz < 3.052:
            return (1.0, -1.433e-3, 7.46e-7, -1.42e-10)  # 2948 MHz
        elif frequency_ghz < 3.180:
            return (1.0, -1.467e-3, 8.05e-7, -1.70e-10)  # 3052 MHz
        elif frequency_ghz < 3.308:
            return (1.0, -1.497e-3, 8.38e-7, -1.80e-10)  # 3180 MHz
        elif frequency_ghz < 3.436:
            return (1.0, -1.504e-3, 8.37e-7, -1.77e-10)  # 3308 MHz
        elif frequency_ghz < 3.564:
            return (1.0, -1.521e-3, 8.63e-7, -1.88e-10)  # 3436 MHz
        elif frequency_ghz < 3.692:
            return (1.0, -1.505e-3, 8.37e-7, -1.75e-10)  # 3564 MHz
        elif frequency_ghz < 3.820:
            return (1.0, -1.521e-3, 8.51e-7, -1.79e-10)  # 3692 MHz
        elif frequency_ghz < 3.948:
            return (1.0, -1.534e-3, 8.57e-7, -1.77e-10)  # 3820 MHz
        else:
            return (1.0, -1.516e-3, 8.30e-7, -1.66e-10)  # 3948 MHz
    
    # C-band (4-8 GHz)
    elif 4.0 <= frequency_ghz < 8.0:
        # Select coefficients based on specific frequency
        if frequency_ghz < 4.180:
            return (1.0, -1.406e-3, 7.41e-7, -1.48e-10)  # 4052 MHz
        elif frequency_ghz < 4.308:
            return (1.0, -1.385e-3, 7.09e-7, -1.36e-10)  # 4180 MHz
        elif frequency_ghz < 4.436:
            return (1.0, -1.380e-3, 7.08e-7, -1.37e-10)  # 4308 MHz
        elif frequency_ghz < 4.564:
            return (1.0, -1.362e-3, 6.95e-7, -1.35e-10)  # 4436 MHz
        elif frequency_ghz < 4.692:
            return (1.0, -1.365e-3, 6.92e-7, -1.31e-10)  # 4564 MHz
        elif frequency_ghz < 4.820:
            return (1.0, -1.339e-3, 6.56e-7, -1.17e-10)  # 4692 MHz
        elif frequency_ghz < 4.948:
            return (1.0, -1.371e-3, 7.06e-7, -1.40e-10)  # 4820 MHz
        elif frequency_ghz < 5.052:
            return (1.0, -1.358e-3, 6.91e-7, -1.34e-10)  # 4948 MHz
        elif frequency_ghz < 5.180:
            return (1.0, -1.360e-3, 6.91e-7, -1.33e-10)  # 5052 MHz
        elif frequency_ghz < 5.308:
            return (1.0, -1.353e-3, 6.74e-7, -1.25e-10)  # 5180 MHz
        elif frequency_ghz < 5.436:
            return (1.0, -1.359e-3, 6.82e-7, -1.27e-10)  # 5308 MHz
        elif frequency_ghz < 5.564:
            return (1.0, -1.380e-3, 7.05e-7, -1.37e-10)  # 5436 MHz
        elif frequency_ghz < 5.692:
            return (1.0, -1.376e-3, 6.99e-7, -1.31e-10)  # 5564 MHz
        elif frequency_ghz < 5.820:
            return (1.0, -1.405e-3, 7.39e-7, -1.47e-10)  # 5692 MHz
        elif frequency_ghz < 5.948:
            return (1.0, -1.394e-3, 7.29e-7, -1.45e-10)  # 5820 MHz
        elif frequency_ghz < 6.052:
            return (1.0, -1.428e-3, 7.57e-7, -1.57e-10)  # 5948 MHz
        elif frequency_ghz < 6.148:
            return (1.0, -1.445e-3, 7.68e-7, -1.50e-10)  # 6052 MHz
        elif frequency_ghz < 6.308:
            return (1.0, -1.422e-3, 7.38e-7, -1.38e-10)  # 6148 MHz
        elif frequency_ghz < 6.436:
            return (1.0, -1.463e-3, 7.94e-7, -1.62e-10)  # 6308 MHz
        elif frequency_ghz < 6.564:
            return (1.0, -1.478e-3, 8.22e-7, -1.74e-10)  # 6436 MHz
        elif frequency_ghz < 6.692:
            return (1.0, -1.473e-3, 8.00e-7, -1.62e-10)  # 6564 MHz
        elif frequency_ghz < 6.820:
            return (1.0, -1.455e-3, 7.76e-7, -1.53e-10)  # 6692 MHz
        elif frequency_ghz < 6.948:
            return (1.0, -1.487e-3, 8.22e-7, -1.72e-10)  # 6820 MHz
        elif frequency_ghz < 7.052:
            return (1.0, -1.472e-3, 8.05e-7, -1.67e-10)  # 6948 MHz
        elif frequency_ghz < 7.180:
            return (1.0, -1.470e-3, 8.01e-7, -1.64e-10)  # 7052 MHz
        elif frequency_ghz < 7.308:
            return (1.0, -1.503e-3, 8.50e-7, -1.84e-10)  # 7180 MHz
        elif frequency_ghz < 7.436:
            return (1.0, -1.482e-3, 8.19e-7, -1.72e-10)  # 7308 MHz
        elif frequency_ghz < 7.564:
            return (1.0, -1.498e-3, 8.22e-7, -1.66e-10)  # 7436 MHz
        elif frequency_ghz < 7.692:
            return (1.0, -1.490e-3, 8.18e-7, -1.66e-10)  # 7564 MHz
        elif frequency_ghz < 7.820:
            return (1.0, -1.481e-3, 7.98e-7, -1.56e-10)  # 7692 MHz
        elif frequency_ghz < 7.948:
            return (1.0, -1.474e-3, 7.94e-7, -1.57e-10)  # 7820 MHz
        else:
            return (1.0, -1.448e-3, 7.69e-7, -1.51e-10)  # 7948 MHz
    
    # X-band (8-12 GHz)
    elif 8.0 <= frequency_ghz < 12.0:
        # Since X-band is quite consistent, use average values for 4 ranges
        if frequency_ghz < 9.0:
            return (1.0, -1.403e-3, 7.17e-7, -1.35e-10)  # Average for 8-9 GHz
        elif frequency_ghz < 10.0:
            return (1.0, -1.410e-3, 7.34e-7, -1.42e-10)  # Average for 9-10 GHz
        elif frequency_ghz < 11.0:
            return (1.0, -1.405e-3, 7.25e-7, -1.37e-10)  # Average for 10-11 GHz
        else:
            return (1.0, -1.392e-3, 6.92e-7, -1.21e-10)  # Average for 11-12 GHz
    
    # Ku-band (12-18 GHz)
    elif 12.0 <= frequency_ghz < 18.0:
        # Ku band is consistent, use average values for 6 ranges
        if frequency_ghz < 13.0:
            return (1.0, -1.394e-3, 7.16e-7, -1.36e-10)  # Average for 12-13 GHz
        elif frequency_ghz < 14.0:
            return (1.0, -1.393e-3, 7.16e-7, -1.37e-10)  # Average for 13-14 GHz
        elif frequency_ghz < 15.0:
            return (1.0, -1.394e-3, 7.19e-7, -1.38e-10)  # Average for 14-15 GHz
        elif frequency_ghz < 16.0:
            return (1.0, -1.400e-3, 7.31e-7, -1.45e-10)  # Average for 15-16 GHz
        elif frequency_ghz < 17.0:
            return (1.0, -1.415e-3, 7.46e-7, -1.48e-10)  # Average for 16-17 GHz
        else:
            return (1.0, -1.448e-3, 8.00e-7, -1.70e-10)  # Average for 17-18 GHz
    
    # K-band (18-26.5 GHz)
    elif 18.0 <= frequency_ghz < 26.5:
        # K-band values sampled at 4 frequency ranges
        if frequency_ghz < 20.0:
            return (1.0, -1.428e-3, 7.71e-7, -1.59e-10)  # Average for 19 GHz
        elif frequency_ghz < 22.0:
            return (1.0, -1.441e-3, 7.91e-7, -1.66e-10)  # Average for 21 GHz
        elif frequency_ghz < 24.0:
            return (1.0, -1.407e-3, 7.31e-7, -1.40e-10)  # Average for 23 GHz
        else:
            return (1.0, -1.414e-3, 7.36e-7, -1.40e-10)  # Average for 25 GHz
    
    # Ka-band (26.5-40 GHz)
    elif 26.5 <= frequency_ghz < 40.0:
        # Ka-band values sampled at 4 frequency ranges
        if frequency_ghz < 30.0:
            return (1.0, -1.450e-3, 7.80e-7, -1.55e-10)  # Average for 28-29 GHz
        elif frequency_ghz < 34.0:
            return (1.0, -1.415e-3, 7.35e-7, -1.40e-10)  # Average for 31-32 GHz
        elif frequency_ghz < 36.0:
            return (1.0, -1.430e-3, 7.50e-7, -1.45e-10)  # Average for 34-35 GHz
        else:
            return (1.0, -1.425e-3, 7.45e-7, -1.40e-10)  # Average for 37-38 GHz
    
    # Q-band (40-50 GHz)
    elif 40.0 <= frequency_ghz < 50.0:
        # Q-band values sampled at 2 frequency ranges
        if frequency_ghz < 42.0:
            return (1.0, -1.460e-3, 7.80e-7, -1.52e-10)  # Average for 41 GHz
        else:
            return (1.0, -1.430e-3, 7.45e-7, -1.40e-10)  # Average for 43 GHz
    
    else:
        # Default case, should not be reached with valid VLA data
        raise ValueError(f'Frequency {frequency_ghz} GHz not in known JVLA bands.')

def jvla_primary_beam_model(frequency_ghz, radius_arcmin):
    """
    Calculate the JVLA primary beam model at a given frequency and radius.
    Uses polynomial coefficients from the EVLA Memo 195.
    
    Parameters:
    frequency_ghz (float): Frequency in GHz
    radius_arcmin (numpy.ndarray): Radius in arcminutes
    
    Returns:
    numpy.ndarray: Primary beam correction factor
    """
    coeffs = get_jvla_beam_coefficients(frequency_ghz)
    a0, a2, a4, a6 = coeffs
    
    # In JVLA model, the polynomial is in terms of (r * f)
    r_norm = radius_arcmin * frequency_ghz
    
    # Compute correction using the polynomial formula
    # Note: In JVLA memo they use 1 + a2*r^2 + a4*r^4 + a6*r^6 format
    correction = a0 + a2 * r_norm**2 + a4 * r_norm**4 + a6 * r_norm**6
    
    return correction

def correct_fits_with_jvla_beam(input_fits, output_fits, beam_threshold=0.01):
    """Apply JVLA primary beam correction to a FITS file."""
    print(f"{MAGENTA}Developed by Arpan Pal at NCRA-TIFR in 2024.{RESET}")
    print(f"{GREEN}Beam Credits Rick Perley, JVLA Memo 195{RESET}")
    print(f"{CYAN}Starting the attack !!!!{RESET}")
    print(f"{YELLOW}Gathering relevant information from the FITS image......................{RESET}")
    print(f"{CYAN}Starting JVLA beam correction!{RESET}")

    # Get data and clean header using flatten_cube
    header, data = flatten_cube(input_fits)
    
    # Handle frequency detection based on data dimensionality
    if data.ndim == 2:
        # For 2D images, use the original find_freq
        frequency_hz = find_freq(header)
        if frequency_hz is None:
            raise ValueError('Frequency information (FREQ) not found in FITS header')
        frequencies_ghz = [frequency_hz/1e9]  # Convert to GHz
        print(f"{GREEN}Found frequency: {frequencies_ghz[0]:.2f} GHz{RESET}")
    else:
        # For cubes, use get_frequencies
        frequencies = get_frequencies(header)
        if not frequencies:
            raise ValueError('No frequency information found in FITS header')
        frequencies_ghz = [f/1e9 for f in frequencies]
        print(f"{GREEN}Found {len(frequencies_ghz)} frequency planes{RESET}")

    # Calculate pixel scale (in arcmin)
    pixel_scale_deg = abs(header['CDELT1'])  # Degrees per pixel
    pixel_scale_arcmin = pixel_scale_deg * 60  # Arcminutes per pixel
    print(f"{GREEN}Pixel scale: {pixel_scale_arcmin:.4f} arcmin/pixel{RESET}")
    
    # Generate radius grid (in arcmin)
    y, x = np.indices(data.shape[-2:])  # Use last two dimensions for spatial coords
    x_center = header['CRPIX1'] - 1
    y_center = header['CRPIX2'] - 1
    r = np.sqrt((x - x_center)**2 + (y - y_center)**2) * pixel_scale_arcmin

    # Create output array with same shape as input
    corrected_data = np.zeros_like(data)

    # Handle both 2D and cube cases
    if data.ndim == 2:
        # Single image case
        freq_ghz = frequencies_ghz[0]
        print(f"{GREEN}Processing single image at {freq_ghz:.3f} GHz{RESET}")
        
        # Get beam correction
        beam = jvla_primary_beam_model(freq_ghz, r)
        print(f"{GREEN}Calculated JVLA Beam!{RESET}")

        with np.errstate(divide='ignore', invalid='ignore'):
            corrected_data = np.where(np.abs(beam) >= beam_threshold, data / beam, 0)
        print(f"{GREEN}JVLA Beam applied to input FITS image!{RESET}")

    else:
        # Cube case
        num_planes = len(frequencies_ghz)
        print(f"{GREEN}Processing cube with {num_planes} frequency planes{RESET}")

        for i in range(num_planes):
            freq_ghz = frequencies_ghz[i]
            print(f"{CYAN}Processing plane {i+1}/{num_planes} at {freq_ghz:.3f} GHz{RESET}")
            
            try:
                # Get JVLA beam model
                beam = jvla_primary_beam_model(freq_ghz, r)
                
                if data.ndim == 3:
                    plane = data[i, :, :]
                else:  # 4D case
                    plane = data[0, i, :, :]
                    
                with np.errstate(divide='ignore', invalid='ignore'):
                    corrected_plane = np.where(np.abs(beam) >= beam_threshold,
                                           plane / beam, 0)
                
                if data.ndim == 3:
                    corrected_data[i, :, :] = corrected_plane
                else:  # 4D case
                    corrected_data[0, i, :, :] = corrected_plane
                    
                print(f"{GREEN}JVLA Beam applied to plane {i+1}!{RESET}")
                
            except ValueError as e:
                print(f"{RED}Warning: Skipping frequency {freq_ghz} GHz - {str(e)}{RESET}")
                # Copy original data for skipped planes
                if data.ndim == 3:
                    corrected_data[i, :, :] = data[i, :, :]
                else:  # 4D case
                    corrected_data[0, i, :, :] = data[0, i, :, :]

    # Read original header to preserve all metadata
    with fits.open(input_fits) as f:
        original_header = f[0].header.copy()
    # Update header history
    original_header['HISTORY'] = f"JVLA beam correction applied {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

    # Write corrected data
    fits.writeto(output_fits, corrected_data, original_header, overwrite=True, output_verify='ignore')
    print(f"{GREEN}JVLA primary beam corrected FITS {'cube' if data.ndim > 2 else 'image'} saved to {output_fits}{RESET}")
    print(f"{MAGENTA}JVLA beam correction complete!{RESET}")


def main():
    parser = argparse.ArgumentParser(description='Apply primary beam correction to a FITS image.')
    parser.add_argument('input_image', type=str, help='Path to the input FITS image file')
    parser.add_argument('output_image', type=str, help='Path to the output FITS image file')
    parser.add_argument('--jvla', action='store_true', help='Use JVLA beam coefficients from EVLA Memo 195')
    parser.add_argument('--beam-threshold', type=float, default=0.01, 
                        help='Beam threshold below which pixels are set to 0 (default: 0.01)')
    args = parser.parse_args()
    
    if args.jvla:
        # Use JVLA-specific beam correction
        correct_fits_with_jvla_beam(args.input_image, args.output_image, beam_threshold=args.beam_threshold)
    else:
        # Use default (GMRT) beam correction
        correct_fits_with_primary_beam(args.input_image, args.output_image, beam_threshold=args.beam_threshold)

if __name__ == "__main__":
    main()