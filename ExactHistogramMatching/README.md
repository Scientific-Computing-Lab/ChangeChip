# Exact Histogram Specification
This is a Python implementation of *Exact Histogram Specification* by *Dinu Coltuc et al.*

In contrast to traditional histogram matching algorithms which only approximate a reference histogram,
this technique can match the exact reference histograms.
This is accomplished by using several kernels which calculate the average of a neighbourhood.
Thereby a pixel can not only be sorted after its value, but also after its average values in more than one neighbourhood.
This helps to create a truely bijective function which is a prerequisite for exact histogram matching.

More information can be found in the [original paper](https://www.researchgate.net/publication/7109912_Exact_Histogram_Specification) or
in Digital Image Processing, 4th Edition, chapter 3.3 which describes the algorithm more concise.
