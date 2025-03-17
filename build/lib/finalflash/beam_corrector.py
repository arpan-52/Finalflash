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



def main():
    parser = argparse.ArgumentParser(description='Apply primary beam correction to a FITS image.')
    parser.add_argument('input_image', type=str, help='Path to the input FITS image file')
    parser.add_argument('output_image', type=str, help='Path to the output FITS image file')
    args = parser.parse_args()

    correct_fits_with_primary_beam(args.input_image, args.output_image)

if __name__ == "__main__":
    main()
