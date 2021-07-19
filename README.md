# ChangeChip
<img align="right" width="170" src="CC_logo.png">

*ChangeChip* was developed to detect changes between an inspected PCB image and a reference (golden) PCB image, in order to detect defects in the inspected PCB.\
The system is based on Image Processing, Computer Vision and Unsupervised Machine Learning.\
*ChangeChip* is targeted to handle optical images, and also radiographic images, and may be applicable to other technologies as well.\
We note that *ChangeChip* is not limited to PCBs only, and may be suitable to other systems that require object comparison by their images.
The workflow of *ChangeChip* is presented as follows:

<img align="center" width="250" height="" src="workflow.PNG">

## Requerments:
- Download the DEXTR model (for the optinal cropping stage):
```
cd DEXTR/models/
chmod +x download_dextr_model.sh
./download_dextr_model.sh
cd ../..
```
- Conda Requerments:
```
conda install pytorch torchvision -c pytorch
conda install matplotlib opencv pillow scikit-learn scikit-image
AND MORE
```
## Running:

# CD-PCB
As part of this work, a small dataset of 20 pairs of PCBs images was created, with annotated changes between them. This dataset is proposed for evaluation of change detection algorithms in the PCB Inspection field.
