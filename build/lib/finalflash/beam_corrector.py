from astropy.io import fits
import numpy as np
import argparse

# Define polynomial coefficients for each band
POLY_COEFFS = {
    'Band-2': (-3.089, 39.314, -23.011, 5.037),
    'Band-3': (-3.129, 38.816, -21.608, 4.483),
    'Band-4': (-3.263, 42.618, -25.580, 5.823),
    'Band-5': (-2.614, 27.594, -13.268, 2.395),
}

def find_freq(header):
    """
    Find frequency value in the FITS header.
    """
    # Try to find frequency in common places
    for i in range(5):
        ctype_key = 'CTYPE%i' % i
        crval_key = 'CRVAL%i' % i
        if ctype_key in header and 'FREQ' in header[ctype_key]:
            return header.get(crval_key)
    
    # If not found, look for specific keywords
    freq = header.get('RESTFRQ') or header.get('FREQ')
    if freq:
        return freq
    
    # If nothing found, return None
    return None

def flatten(filename, channel=0, freqaxis=0):
    """Flatten a FITS file to create a 2D image. Returns new header and data."""
    from astropy.wcs import WCS

    with fits.open(filename) as f:
        naxis = f[0].header['NAXIS']
        if naxis < 2:
            raise ValueError('Cannot make map from this FITS file.')
        if naxis == 2:
            return f[0].header, f[0].data

        w = WCS(f[0].header)
        wn = WCS(naxis=2)

        wn.wcs.crpix[0] = w.wcs.crpix[0]
        wn.wcs.crpix[1] = w.wcs.crpix[1]
        wn.wcs.cdelt = w.wcs.cdelt[0:2]
        wn.wcs.crval = w.wcs.crval[0:2]
        wn.wcs.ctype[0] = w.wcs.ctype[0]
        wn.wcs.ctype[1] = w.wcs.ctype[1]

        header = wn.to_header()
        header["NAXIS"] = 2
        header["NAXIS1"] = f[0].header['NAXIS1']
        header["NAXIS2"] = f[0].header['NAXIS2']
        copy = ('EQUINOX', 'EPOCH')
        for k in copy:
            r = f[0].header.get(k)
            if r:
                header[k] = r

        dataslice = []
        for i in range(naxis, 0, -1):
            if i <= 2:
                dataslice.append(np.s_[:],)
            elif i == freqaxis:
                dataslice.append(channel)
            else:
                dataslice.append(0)

        header["FREQ"] = find_freq(f[0].header)

        try:
            header["BMAJ"] = f[0].header['BMAJ']
            header["BMIN"] = f[0].header['BMIN']
            header["BPA"] = f[0].header['BPA']
        except KeyError:
            pass

        return header, f[0].data[tuple(dataslice)]

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

def correct_fits_with_primary_beam(input_fits, output_fits, beam_threshold=0.01):
    """Apply primary beam correction to a FITS file."""
    # Flatten the FITS file to a 2D image
    header, data = flatten(input_fits)
    
    # Get frequency information from the header (assuming CRVAL3 holds frequency in Hz)
    frequency_hz = header.get('FREQ')
    if frequency_hz is None:
        raise ValueError('Frequency information (FREQ) not found in the FITS header.')
    frequency_ghz = frequency_hz / 1e9  # Convert to GHz
    print(f"Found frequency: {frequency_ghz} GHz")

    # Determine which band's coefficients to use based on the frequency
    band_coeffs = get_band_coefficients(frequency_ghz)
    print(f"Using polynomial coefficients for band: {band_coeffs}")

    # Calculate the pixel scale (separation per pixel in arcmin)
    pixel_scale_deg = abs(header['CDELT1'])  # Degrees per pixel
    pixel_scale_arcmin = pixel_scale_deg * 60  # Arcminutes per pixel
    print(f"Pixel scale: {pixel_scale_arcmin} arcmin/pixel")
    
    # Generate a grid of pixel positions
    y, x = np.indices(data.shape)
    x_center = header['CRPIX1'] - 1
    y_center = header['CRPIX2'] - 1
    r = np.sqrt((x - x_center)**2 + (y - y_center)**2) * pixel_scale_arcmin  # Radius in arcmin
    
    # Compute the primary beam model for each pixel
    beam = primary_beam_model(frequency_ghz, r, band_coeffs)
    
    # Check where the absolute value of the beam is less than the threshold
    with np.errstate(divide='ignore', invalid='ignore'):
        corrected_data = np.where(np.abs(beam) >= beam_threshold, data / beam, 0)

    # Write the corrected data to a new FITS file
    fits.writeto(output_fits, corrected_data, header, overwrite=True)
    print(f"Primary beam corrected FITS file saved to {output_fits}")

def main():
    parser = argparse.ArgumentParser(description='Apply primary beam correction to a FITS image.')
    parser.add_argument('input_image', type=str, help='Path to the input FITS image file')
    parser.add_argument('output_image', type=str, help='Path to the output FITS image file')
    args = parser.parse_args()

    correct_fits_with_primary_beam(args.input_image, args.output_image)

if __name__ == "__main__":
    main()