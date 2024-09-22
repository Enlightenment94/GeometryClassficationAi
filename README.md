Target of learning model: Classification of geometry on an unsteady background.

Approach:

Build a simple MLP - bad results.
Reduce size - better results.
Use CNN for classification on a white background.
Poor results on a disturbed background.
Perform segmentation with a mask.
-> Extract the predicted mask.
Build a model to predict figure labels on the extracted predicted mask.

Arrow is current step