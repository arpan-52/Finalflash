from astropy.io import fits
import numpy as np
import argparse
import logging
import os

# Define the Polynomial Coefficients as per your bands
POLY_COEFFS = {
    'Band-2': (-3.089, 39.314, -23.011, 5.037),
    'Band-3': (-3.129, 38.816, -21.608, 4.483),
    'Band-4': (-3.263, 42.618, -25.580, 5.823),
    'Band-5': (-2.614, 27.594, -13.268, 2.395),
}

def flatten(filename, channel=0, freqaxis=0):
    """ Flatten a fits file so that it becomes a 2D image. Return new header and data """
    from astropy.wcs import WCS as pywcs

    f = fits.open(filename)
    naxis = f[0].header['NAXIS']
    if naxis < 2:
        raise ValueError('Cannot make map from this FITS file.')
    if naxis == 2:
        return f[0].header, f[0].data

    w = pywcs(f[0].header)
    wn = pywcs(naxis=2)

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

def find_freq(header):
    """
    Find frequency value in most common places of a fits header
    """
    for i in range(5):
        type_s = header.get('CTYPE%i' % i)
        if type_s is not None and type_s.startswith('FREQ'):
            return header.get('CRVAL%i' % i)
    return None

def apply_beam_correction(input_image, output_image):
    """Apply beam correction to the input FITS file and save the result to the output file."""
    # Load input image
    hdr, data = flatten(input_image)
    
    # Find frequency
    freq = find_freq(hdr)
    if freq is None:
        raise ValueError('Frequency information not found in the header.')

    # Determine the band
    band = get_band(freq)
    if band not in POLY_COEFFS:
        raise ValueError(f'Band for frequency {freq} MHz not recognized.')
    
    # Get polynomial coefficients for the band
    a, b, c, d = POLY_COEFFS[band]
    
    # Apply beam correction
    corrected_data = apply_correction(data, a, b, c, d, freq)
    
    # Save the corrected FITS file with the same header
    fits.writeto(output_image, corrected_data, hdr, overwrite=True)
    print(f"Beam corrected FITS file saved to {output_image}")

def get_band(freq):
    """Determine the band based on frequency."""
    if 125 <= freq < 250:
        return 'Band-2'
    elif 250 <= freq < 500:
        return 'Band-3'
    elif 550 <= freq < 850:
        return 'Band-4'
    elif 1050 <= freq < 1450:
        return 'Band-5'
    else:
        raise ValueError('Frequency is out of range for known bands.')

def apply_correction(data, a, b, c, d, freq):
    """Apply beam correction using polynomial coefficients."""
    x = np.arange(data.shape[0])  # Example x-values; adjust as needed
    correction = 1 + (a / 1e3) * x**2 + (b / 1e7) * x**4 + (c / 1e10) * x**6 + (d / 1e13) * x**8
    return data * correction

def main():
    parser = argparse.ArgumentParser(description='Apply beam correction to FITS images.')
    parser.add_argument('input_image', type=str, help='Path to the input FITS image file')
    parser.add_argument('output_image', type=str, help='Path to the output FITS image file')
    args = parser.parse_args()

    apply_beam_correction(args.input_image, args.output_image)

if __name__ == "__main__":
    main()
