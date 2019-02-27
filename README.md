# dig_indicator

### Instructions:

In data folder, it includes the pre-trained fasttext bin model(vector length: 20), pre-trained incall, outcall, movement, multi-girls, risky models.

### test_file.py

When use test_file.py to test your file. Following the command:    python test_file.py input_file output_file
In output file, the sentence in each line is followed by the json object, which includes the probability for each attribute like incall, outcall, risky. You can also refer to the input_sample file and output_sample file.


### test.py

In test.py file, first it outputs the sentence vector of user's input, then outputs the category and its probability.

Sample:

Input: incall girl

[[ 0.03920003  0.34195638  0.2633262  -0.33317611  0.23232044  0.19340721
   1.29433036  0.2046183  -0.28624511 -0.53930861  0.16153046  0.14273527
  -0.4219546  -0.3338438  -0.53146815 -0.20583555 -0.67541367 -0.05810158
   0.6235007   0.55164051]]
   
[{'score': 0.9999998981559567, 'value': 'incall'},

{'score': 1.0, 'value': 'outcall'}, 

{'score': 0.5521221254944197, 'value': 'movement'}, 

{'score': 1.2153211397728898e-07, 'value': 'multi_girls'}, 

{'score': 1.0540844377029096e-41, 'value': 'risky_activity'}]

