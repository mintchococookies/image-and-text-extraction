# Image and Text Extraction using OpenCV and PyTesseract

Python program to extract engineering drawings from a group of images and extract all the tabular data into a single Excel spreadsheet.

### Sample input
Sample input image containing tabular data and an engineering drawing (all input images can be found in the input-dataset folder)
![Alt Text](https://github.com/mintchococookies/image-and-text-extraction/raw/main/input-dataset/20.png)

### Process
Mask generated using OpenCV erosion and dilation on the arrowheads around the engineering drawing:
![Alt Text](https://github.com/mintchococookies/image-and-text-extraction/raw/main/images/dilation_output.png)

Drawing extracted using the mask (inverted):
![Alt Text](https://github.com/mintchococookies/image-and-text-extraction/blob/main/images/drawing.png)

Tabular data retained after drawing extraction:
![Alt Text](https://github.com/mintchococookies/image-and-text-extraction/blob/main/images/table20.png)

### Output
Extracted image:
![Alt Text](https://github.com/mintchococookies/image-and-text-extraction/blob/main/output/drawing20.png)

Excel spreadsheet compiling all the tabular data extracted using Pytesseract (text-extraction-output.xlxs):
![Alt Text](https://github.com/mintchococookies/image-and-text-extraction/blob/main/images/blueprint-information.PNG)
