Thank you Lori, for the introduction. So once we had concluded our exploration and found the features we believed to be significant we proceeded modeling to see if we could create a machine learning model that could accurately predict whether a case would be resolved early, very early, late, very late, or one time.

Before we could proceed into modeling, however, we first had to establish a baseline model, as this would be our goal to make a machine learning model that could outperform a baseline prediction.

We found our baseline by exploring our target variable, the response times. It was discovered that of all the cases in our data frame 57% of cases were resolved "very early". Which means if you assumed that every 311 case the city of San Antonio handled would be resolved very early, you would be correct about a little over half of the time. While this is great news for the city of San Antonio it provided a lofty goal for us to reach in order to beat baseline.

In our modeling stage we had utilized multiple machine learning algorithms and recorded their performance for comparison. On our training data we found that the best performing models were the decision tree and random forest models, each capable of improving on the baseline by approximately 10%. 

Ultimately we proceeded with the decision tree model as it had performed slightly better on unseen validate data than the random forest which had a slight dip in performance. 

We found that our decision tree model also had the highest recall and precision for both early and late cases, which is important for minimizing errors when setting expectations for resolution time.

Overall the decision tree had a consistent performance on training and unseen data making it our most valuable model.

While we are pleased with this model's performance we hope this model can be improved upon in future iterations.

With that I would like to welcome back Caitlyn, who will share with you our conclusion.