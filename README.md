# spectroscopy-img-proc

## various spectroscopy / image processing code that I use occasionally and need to remind myself

# Echelle processing
  Code to extract an echelle image which consists of multiple diffraction orders over a 2-d array.   This works for UV/Vis spectra 200-1000 nm.  The diffraction orders are automatically calibrated for grating parameters and cross-disperser glass using reference peaks from a mercury argon lamp.  The diffraction orders are extracted using fast numpy strided arrays.

  
