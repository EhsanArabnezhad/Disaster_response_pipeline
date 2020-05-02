# Disaster Response Twitter Message Classification

# App is deployed online on Heroku

## https://ehsan-disaster-response.herokuapp.com/

<p align="center"> 
<img src="https://github.com/ehsanarabnezhad/Disaster_response_pipeline/blob/master/app/static/usa.png", style="width:30%">
</p>

# repository layout

```

├── app
│   ├── run.py
│   ├── static
│   │   ├── chile01.png
│   │   ├── haiti01.png
│   │   ├── pakistan01.png
│   │   └── usa01.jpg
│   └── templates
│       ├── go.html
│       └── master.html
├── data
│   ├── disaster_categories.csv
│   ├── disaster_messages.csv
│   ├── DisasterResponse.db
│   └──process_data.py
│
├── ETL_pipeline_prep
│   ├── categories.csv
│   ├── ETL Pipeline Preparation.ipynb
│   └── messages.csv
│
├── ML_pipeline_prep
│   ├── DisasterResponse.db
│   ├── fair_model__ada_02.pkl
│   ├── fair_model__ada_02.sav
│   └── ML Pipeline Preparation.ipynb
├── models
│   ├── fair_model__ada_02.pkl
│   └── train_classifier.py
├── README.md
├── requirement.txt
│
└── Disaster_response_app
	├── data
    ├── disaster_response.py
    ├── disasterresponseapp
    ├── models
    ├── Procfile
    └── requirements.txt

```
## How to run the code:

0. Clone the repository use: `https://github.com/EhsanArabnezhad/Disaster_response_pipeline.git`, and pip install `requirement.txt`
```
python3 -m venv name_of_your_choosing
source name_of_your_choosing/bin/activate
pip install --upgrade pip
pip install -r requirements.txt                      # install packages in requirement
```
```
### Instructions:
```
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:5000/

```
## Project approach: 
```
After merging and evaluating classes, there are ~20% of data without any positive class (all the lables are 0). On the Figure eight web-site which hosts the data set, it it stated that the classes are binary however the 'related' class had 3 values. I thought I have 2 ways to encounter this 1- To remove 'related' labels == 2 or to convert them into 0. I chose to convert them to 0 since after investigating the messages it seemd that those messages are not related to a Disaster happening around the world. Messages similar to "I hope you are doing fine" !!. By this message also we can notice that it is best not to remove rows with no positive labels as well. Beacaue that sample sentence is not related to any class. 

The class label named "child_alone" doesnt have any positive value so it is removed from data set. Notice that to run the app the very first work is needed to be done is to remove this column from dataset. Otherwise however there will not be any problem in classification but the results will shift one column to the right and will be totally wrong.

The tokenizer is removing URLs in 2 steps. First the pattern of some the URLs in the text contains [space] between http and ":" and "//" so it should be taken care of. Second for the other normal URLs a perfect regex pattern is used to find them. The email addresses also not too much are removed with a regex pattern that I used in another project back in times which proved to be very accurate. After these 2 steps I removed all chars except [a-zA-Z] to clean more. (However this step is too lazy) 
The pipeline with a Tfisfvectorizer and Adaboost seemed to have the best results. 

After evaluting many constructed pipelines I took the best one and fit it with all the data (not splitted) since I think more data will build the better model.

<p align="center"> 
<img src="https://github.com/EhsanArabnezhad/Disaster_response_pipeline/blob/master/app/static/home_page.png" style="width:30%">
</p>

## Deployment on Heroku

It took a will until I managed to read the pickled model correctly online. There was a deployment problem which prompted "model = pickle.load(open('models/fair_model__ada_02.pkl', 'rb'))" . AttributeError: Can't get attribute 'tokenize' on <module '__main_ ' from '/app/.heroku/python/bin/gunicorn'>

I added the following function to resolve this. The reason of this I believe is for the heirarchy complications of pickle and the fact that pickle is created starting from the main() function while on localhost. 
```
class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        if name == 'tokenize':
            return tokenize
        return super().find_class(module, name)

model = CustomUnpickler(open('files/fair_model__ada_02.pkl', 'rb')).load()
```

By using a Custompickler function before loading the app, the class overwrite intrinsic pickle find_class function and provide it a direct address to tokenize function. So simply providing the tokenize function on the top was not enough. Need to also add that for consistency and due to sklearn warning only pickle dump and load are used (instead of joblib). Also see https://stackoverflow.com/questions/27732354/unable-to-load-files-using-pickle-and-multiple-modules/27733727#27733727



