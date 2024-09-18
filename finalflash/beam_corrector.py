import numpy as np
from astropy.io import fits
import argparse

# Define the band coefficients
band_coefficients = {
    "Band-2": {"range": (0.125, 0.250), "coeffs": [-3.089, 39.314, -23.011, 5.037]},
    "Band-3": {"range": (0.250, 0.500), "coeffs": [-3.129, 38.816, -21.608, 4.483]},
    "Band-4": {"range": (0.550, 0.850), "coeffs": [-3.263, 42.618, -25.580, 5.823]},
    "Band-5": {"range": (1.050, 1.450), "coeffs": [-2.614, 27.594, -13.268, 2.395]}
}

# Function to compute the primary beam model for a given frequency and pixel separation (in arcmin)
def primary_beam_model(frequency, separation_arcmin, band_coeffs):
    x = separation_arcmin * frequency
    a, b, c, d = band_coeffs
    beam_value = 1 + (a / 1e3) * x**2 + (b / 1e7) * x**4 + (c / 1e10) * x**6 + (d / 1e13) * x**8
    return beam_value

# Function to determine the frequency band based on the frequency
def get_band_coefficients(frequency_ghz):
    for band, data in band_coefficients.items():
        f_min, f_max = data["range"]
        if f_min <= frequency_ghz <= f_max:
            return data["coeffs"]
    raise ValueError(f"Frequency {frequency_ghz} GHz does not match any known uGMRT band.")

# Function to read, correct, and write the FITS file
def correct_fits_with_primary_beam(input_fits, output_fits):
    # Open the input FITS file
    with fits.open(input_fits) as hdul:
        # Extract image data
        data = hdul[0].data
        header = hdul[0].header
        
        # Get frequency information from the header (assuming CRVAL3 holds frequency in Hz)
        frequency_hz = header['CRVAL3']
        frequency_ghz = frequency_hz / 1e9  # Convert to GHz
        
        # Determine which band's coefficients to use based on the frequency
        band_coeffs = get_band_coefficients(frequency_ghz)
        
        # Calculate the pixel scale (separation per pixel in arcmin)
        pixel_scale_deg = abs(header['CDELT1'])  # Degrees per pixel
        pixel_scale_arcmin = pixel_scale_deg * 60  # Arcminutes per pixel
        
        # Generate a grid of pixel positions
        y, x = np.indices(data.shape)
        x_center = header['CRPIX1'] - 1
        y_center = header['CRPIX2'] - 1
        r = np.sqrt((x - x_center)**2 + (y - y_center)**2) * pixel_scale_arcmin  # Radius in arcmin
        
        # Compute the primary beam model for each pixel
        beam = primary_beam_model(frequency_ghz, r, band_coeffs)
        
        # Apply the primary beam correction (divide by the beam)
        corrected_data = data / beam
        
        # Write the corrected data to a new FITS file
        hdul[0].data = corrected_data
        hdul.writeto(output_fits, overwrite=True)

# CLI for the script
def main():
    parser = argparse.ArgumentParser(description="Correct a FITS image using uGMRT primary beam model")
    parser.add_argument('input_fits', type=str, help='Path to the input FITS file')
    parser.add_argument('output_fits', type=str, help='Path to save the corrected FITS file')

    args = parser.parse_args()

    # Perform the primary beam correction
    correct_fits_with_primary_beam(args.input_fits, args.output_fits)

if __name__ == "__main__":
    main()
