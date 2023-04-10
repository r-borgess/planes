# planes
A plane detection system based on Affine Homography calculation.

- Image processing
- Color space conversion and channel splitting
- Feature detection and Feature matching (SIFT)
- Delaunay triangulation
- Homography calculation
- Clustering (k-means)
- Plane detection

## Tools

- Python 3.10
  - OpenCV 2
  - Numpy
  - Matplotlib
  - Scipy
  - Sklearn
  
 ## Input
 
 Two consectutive frames (png, jpg, etc) defined in the code.
 
 ## Output
 
 - Original frames
 - Color space conversion
 - Feature detection results
 - Delaunay triangulation results
 - Homography transformation
 - Clustering results
 
 ## Execution
 
 `python final.py`
 
 Or press run on your favorite IDE.
 
## Notes
 
### Color information

Instead of using black and white frames, the program uses a combination of the luminance (Y) and both crominance (U and V) channels of the frame in order to obtain more accurate flow calculation by using color and intensity information combined. The combination of the channels can be modified in the code. It is also possible to use the RGB format or even the bw version.

### Future

I would like to turn this project into a more robust system, maybe towards 3D reconstruction. Feel free to contribute.
