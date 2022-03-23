# RQD-estimation
The base models are from https://github.com/qubvel/segmentation_models 

pip install -U segmentation-models==0.2.1

Then you can run

python train.py

After the model training

Change the model name in predict.py on line 22

You can run 

python predict.py

Then you get the segmentation result using single model

Using the model ensemble

python imgensemble.py

The segmentation result using ensemble model can be obtained

For all the codes, we have provided the sample data to test. Only need to change the directory.

Project is distributed under MIT Licence
