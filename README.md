# dss_crf

The original version of this code can be found [here](https://github.com/lucasb-eyer/pydensecrf.git).
Notice that please provide a link to this project as a footnote or a citation if you plan to use it.

***This code is meant to be used as a submodule for other saliency projects, such as [Saliency](https://github.com/Dom303/Saliency), [Deep-GDM](https://github.com/Dom303/Deep-GDM) and [HED-GDM](https://github.com/Dom303/HED-GDM).***

## Windows setup
Open a command prompt in the folder containing the dss_crf project with **Administrator rights**.
Run the command:
```
python setup.py install
```

### Usage
Run the file **run_densecrf_folder.py** tu run the denseCRF on all the saliency files from a specific folder. Change the following parameters according to the comments:
- base_folder
- image_folder
- saliency_folder
- output_folder

## Linux setup
### Setup
```
sudo python setup.py install
```  

### Usage
```
cd examples
python dense_hsal.py im1.png anno1.png out1.png
```
im1.png -> source image  
anno1.png -> predicted saliency map (should be gray level)  
out1.png -> output (after CRF)  


## Packages

The following packages are required for the installation to complete (but they are not all listed). If the installation is not successful, install the missing packages. They can be installed via the **pip install** command.

- numpy
- cython
- ensurepip
- scipy
- sys
- cv2
- skimage
- glob
- os
- joblib
- multiprocessing
