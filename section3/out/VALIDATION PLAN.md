# VALIDATION PLAN




### Intended Use

To provide an end-to-end system consisting of an algorithm that measures hippocampal volumes in studies received by clinical imaging archives, and integrates with a clinical viewer.

### Training Data Collection

The 'Hippocampus' dataset from the 'Medical Decathalon' competition is used for training. Original images consist of T2 MRI brain scans. Dataset consists of cropped volumes including only areas of interest

### Training Data Labels

Labels consist of masks verified by radiologists 

### Algorithm Evaluation

Training performance of the algorithm was measured using the Dice and Jaccard scoring indeces. Real world performance will be measured using the above indeces, as well as additional criteria such as sensitivity and specficity, which will be rattified by radiologists.


### Ideal Usecase Data
The algorithm will perform well on  T2 MRI images of the brain which of been cropped to inlcude mostly the Hippocampus and the area around it. The algorithm might not perform well with images that consist of whole brain scans, without the area of concern focused upon














   
