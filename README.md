# AGAI

This research project aims at single classifying a leaf between 5 classes: necrosis, chlorosis, residue, dirt, and water. The system comprises of rf-detr, an object detection model, used for identifying the most "prominent" leaf in a picture that may consist of multiple leaves. Analyzing a singular leaf is a much more simple task that improves classification accuracy. After rf-detr, the image is then cropped by the bounding box produced by rf-detr and then some padding(100px). This image is then resized into a 256x256 px image and passed onto the FastViT+MLP classifier.

Data Collection & Training:

This research used 4200 images total across the 5 classes for training, validation, and testing. The model weights provided for the classifier is the weights that were finalized from a 70/15/15 data split trial. This means 420 images per class, and in each class 294 images were used for training, 63 images were used for validation, and 63 images were used for testing. These model weights achieved a 99% accuracy in testing.


Final_Traning_FastViT.ipynb is the notebook with the training for the FastViT + MLP classifier only. It is trained on preprocessed data of images that have ALREADY BEEN CROPPED by rf-detr. See the rf-detr.ipynb notebook to see the training used for rf-detr. Below is an example of how rf-detr modifies each image. 


<p align="center">
<img src="data-preprocessing.png" alt="Examples" width="70%">
</p>

After this, more image preprocessing occurs, these images get resized to 256x256 images. All 4200 images were preprocessed like this and then finally trained on with the FastViT+MLP classifier. Examples of the classification is below.

<p align="center">
<img src="example_classification.png" alt="Examples" width="70%">
</p>
