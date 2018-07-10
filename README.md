# AWS Serverless inference for ML models

This code is a Cloud9 Code that implements Lambda functions and API Gateway for model inference. This is a sample code for image classification:

Here is the breakdown of the Lambda Functions:

**imageProcessing:**

This function will process the image before and convert it to numpy array after reshaping it. The numpy representation of the image will be uploaded to S3.


**prediction**

This will download the image numpy representation from S3 and load the model then make predictions. It will send the prediction results to the `imageProcessing` function then to the API Gateway response.


Each of the function folders contain a .yaml file that represents the cloudformation template that will create the function and the API Gateway associated.
