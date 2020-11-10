# Tensorflow-MortgageModel
Model that predicts if a user will repay their loans or not

## How model is built
Model is trained off of a data set provided by Kaggle 
https://www.kaggle.com/wordsforthewise/lending-club

Model is created to determine if an individual taking out a loan will repay their loan or not.
This project is an example of supervised learning as a dataset was provided to the model and it was trained based of different charactericts of people, such as:
- Job
- Year's of employment
- Their current grade/sub grade (similiar to a credit score)
- Number of accounts
- Number of mortgage accounts they currently have

## Confusion Matrix
<img width="480" alt="Screen Shot 2020-11-09 at 10 14 52 PM" src="https://user-images.githubusercontent.com/69999501/98630917-6e608900-22d9-11eb-93ff-932c8d7fc1c7.png">


- 89% accuracy --> Isn't good accuracy considering the model would already has a 80% accuracy since its an unbalanced dataset.
- 98% precision
- 44% recall 
- 61% f1-score

## Model results with test data
<img width="318" alt="Screen Shot 2020-11-09 at 10 26 16 PM" src="https://user-images.githubusercontent.com/69999501/98631433-a7e5c400-22da-11eb-91ad-b456d4833ff3.png">

Model has successfully determined the outcome of the test set
