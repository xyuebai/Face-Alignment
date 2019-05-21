# Face-Alignment
Face alignment with 5 points (nose, center of both eyes, both corners of mouth), this project include model training, landmarks detection and alignment based on the detected landmarks and a 224 by 224 symmetric 5 points facial template.


### Prerequisites
* Anaconda 3.6
* dlib
* opencv (conda)

### Little instruction
* landmark_detector_trainer.py
_To train the 5 landmark detector, a trained model is in the folder  `landmark_5_model`.

* face_aligner.py
_Does the actual facial alignment work.

### Examples
![Example 1: Before aligning](example_output/flickr_2_image16315_2.jpg)
![Example 1: After aligning](example_output/flickr_2_image16315.jpg)
![Example 2: Before aligning](example_output/flickr_3_image37566_2.jpg )
![Example 2: After aligning](example_output/flickr_3_image37566.jpg )
