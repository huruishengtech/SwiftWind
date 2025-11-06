# SwiftWind
Code for paper: "SwiftWind: Sparse-to-full wind speed reconstruction from noisy and partial observations with flexible arbitrary-location inference". SwiftWind is a flexible deep learning framework that integrates sparse real-world observations with coarse background fields to reconstruct continuous sea surface wind speed fields. SwiftWind adopts an encoder-decoder architecture that encodes arbitrary numbers of non-gridded observations into a latent representation, using latitude-longitude positional encodings and mask-based attention mechanisms to support variable-length sequences. The model decodes wind speeds at arbitrary query locations, leveraging learnable latent arrays as query matrices in both encoder and decoder to capture spatial correlations and underlying wind structure patterns.

# Architecture
<p align="center">
  <img src="https://raw.githubusercontent.com/huruishengtech/SwiftWind/main/model_architecture.png" width="700" alt="SwiftWind Architecture Diagram">
</p>

# Code acknowledgements
We acknowledge that this work builds upon the open-source implementations of
[Senseiver](https://github.com/OrchardLANL/Senseiver) and [Perceiver IO](https://github.com/krasserm/perceiver-io)).
We also make extensive use of PyTorch, NumPy, Matplotlib, and PyTorch Lightning.
We sincerely thank the original authors for their contributions to the open-source community.

# Data availability
The datasets used in this study are publicly available from the following sources:
(i) The GFS 6-hour forecast data can be obtained from the National Center for Atmospheric Research (NCAR) Research Data Archive at https://rda.ucar.edu/datasets/ds084.1/. (ii) The ERA5 reanalysis dataset is available from the Copernicus Climate Data Store (CDS) at https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-timeseries.
 (iii) The ICOADS observations can be accessed at https://www.ncei.noaa.gov/products/international-comprehensive-ocean-atmosphere-data-set. (iv) The ASCAT scatterometer data is available from  https://scatterometer.knmi.nl/. 

# Requirements
* Python 3.9.23
* pytorch-lightning 2.0.0
* torch 2.0.1+cu118
* scipy 1.10.0
* numpy 1.26.4
* netCDF4 1.6.2
* pandas 2.0.0
* xarray 2023.6.0
