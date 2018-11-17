# xray-super-res
Supervised x-ray image super-resolution and denoising. 

The scripts here expect the following folder structure:

xray_images/
- test_images_64x64/
- test_images_output/
- train_images_64x64/
- train_images_128x128/
- train_images_output/

All but the "output" folders can be obtained by extracting the zipped data set. 

As stated in the project specs, the submission script is called as follows:
```python
python3 sample_request.py --netid NETID --token CRAZY_LONG_TOKEN --image-dir RELATIVE_PATH_TO_IMAGE_DIR_NO_QUOTES --server SERVER_IP
```

The Python file "test_script.py" simply repeats each pixel in the original image four times to get a "high-res" version. We can use this as our weakest baseline. 