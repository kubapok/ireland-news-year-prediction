# Ireland news headlines

# Dataset source and thanks

Predict the year 
Start Date: 1996-01-01 End Date: 2019-12-31


Dataset taken from https://www.kaggle.com/therohk/ireland-historical-news on 19.06.2020.
Special thanks to Rohit Kulkarni who created it.

You may find whole dataset (including the test dataset) in the link above.
The dataset in the link may be updated.
Please, do not incorporate any of the data from this kaggle dataset (or others) to your submission in this gonito challange.

Data is exactly the same as in ireland news headlines

## Context (from https://www.kaggle.com/therohk/ireland-historical-news )

This news dataset is a composition of 1.48 million headlines posted by the Irish Times operating within Ireland.

Created over 160 years ago; the agency can provides long term birds eye view of the happenings in Europe.


# Challange creation

Year is normalized as follows:

'''
    days_in_year = 366 if is_leap else 365
    normalized = d.year + ((day_of_year-1)  / days_in_year)
'''

train, dev, test split is 80%, 10%, 10% randomly

note that there are very similar headlines in the data

I did not make any effort to prevent from going one sentence like this to the train and second one to the test.

I used a first category in the classification task. E.g there is "world" instead of "world.us" as on original dataset.

