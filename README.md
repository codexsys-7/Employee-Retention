# Employee-Retention

# import all the libraries required for the project 
# Ex:numpy,pandas,sweetviz,seaborn,matplotlib,sklearn,imblearn,ipywidgets and (scikitlearn) if neceessary

#!pip install sweetviz
#!pip install ppscore
#!pip install imblearn
#!pip install ipywidgets

#numpy for numerical calculations.
import numpy
#pandas for data manipulation and analysis.
import pandas

#for exploratary data analysis.
import sweetviz

#for data visualisations
import seaborn
import matplotlib
import ppscore as pps

import sklearn
import imblearn

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#for interactive console
import ipywidgets
import ipywidgets as widgets
from ipywidgets import interact
from ipywidgets import interact_manual

#setting up size of the figures
plt.rcParams['figure.figsize']=(16,5)

#setting up style of the plots.
#for more styles(plt.style.available)
plt.style.use('fivethirtyeight')


#READING THE DATASETS.
#Reading the data from the .csv(comma seperated variables) files.
#Train.csv to train the model.
#after learning the patterns from the Testing datasets, we then predict the TARGET variable.
#Test.csv to test the model
dodge=pd.read_csv("Train.csv")
charger=pd.read_csv("Test.csv")

#returns first 5 entries in the Train.csv file.
#dodge.head(6), return first 6 entries.
dodge.head()

#returns first 5 entries from Test.csv file.
charger.head()

dodge.tail()

charger.tail()

#shape of the Dataset
#this statement returns the number of entries present in the Train.csv and Test.csv files.
#size of the data:(no_of_entries,no_of_columns)

print("Size of the Training Data:",dodge.shape)
print("Size of the Test Data:",charger.shape)

# <center>Data Description</center>

<table>
    <tr>
        <td><b>Variable</b></td>
        <td><b>Definition</b></td>
    </tr>
    <tr>
        <td>employee_id</td>
        <td>Unique ID for employee<td>
    </tr>
    <tr>
        <td>department</td>
        <td>Department of employee</td>
    </tr>
    <tr>
        <td>region</td>
        <td>Region of employment (unordered)</td>
    </tr>
    <tr>
        <td>education</td>
        <td>Education Level</td>
    </tr>
    <tr>
        <td>gender</td>
        <td>Gender of Employee</td>
    </tr>
    <tr>
        <td>recruitment_channel</td>
        <td>Channel of recruitment for employee</td>
    </tr>
    <tr>
        <td>no_of_trainings</td>
        <td>no of other trainings completed in previous year on soft skills, technical skills etc.</td>
    </tr>
    <tr>
        <td>age</td>
        <td>Age of Employee</td>
    </tr>
    <tr>
        <td>previous_year_rating</td>
        <td>Employee Rating for the previous year</td>
    </tr>
    <tr>
        <td>length_of_service</td>
        <td>Length of service in years</td>
    </tr>
    <tr>
        <td>KPIs_met >80%</td>
        <td>if Percent of KPIs(Key performance Indicators) >80% then 1 else 0</td>
    </tr>
    <tr>
        <td>awards_won?</td>
        <td>if awards won during previous year then 1 else 0</td>
    </tr>
    <tr>
        <td>avg_training_score</td>
        <td>Average score in current training evaluations</td>
    </tr>
    <tr>
        <td>is_promoted	(Target)</td>
        <td>Recommended for promotion</td>
    </tr>
</table>

#Determining the datatype of the columns.  
dodge.info()

charger.info()

#This statement helps to pick our desired style for the figures.
#eg:ln[22]
plt.style.available

#returns the count of employees either promoted or not.
dodge["is_promoted"].value_counts()

#PLOTING TARGET CLASS BALANCE FOR THE TARGET ATTRIBUTE(is_promoted)
#size and style for plotting the graph and pie chart.
plt.rcParams["figure.figsize"]=(15,5)
plt.style.use("Solarize_Light2")



#subplot(no_of_rows(1),no_of_columns(2),position_of_the_chart(1/2))
plt.subplot(1,2,1)
#ploting a graph regarding the target atrribute(is_promoted)
#sns=seaborn data visualisation library.
sns.countplot(dodge["is_promoted"])
#labelling the x axis with promoted/not.
plt.xlabel("promoted/not",fontsize=15)



#ploting a pie chart with respect to is_promoted attribute.
plt.subplot(1,2,2)
#plot kind=pie chart,
#explode=taking a piece of pie out i.e taking (0,0.1)% from the pie out, 
#autopct=decimal points after the number 2.52,6.78 etc (%.1f%% (%% is used to give percent symbol after the decimal points))
#startangle=it can be any angle from 0-90, 
#labels = the names we wanted to give for the pieces of the pie("Not_Promoted_Employees","Promoted_Employees"),
#shadow=to give a shadow to the pie at the bottom(True-shadow out,False-no shadow), 
#pctdistance=distance between the points. 
dodge["is_promoted"].value_counts().plot(kind="pie",explode=[0,0.2],
                                         autopct="%.1f%%",
                                         startangle=90,
                                         colors=["magenta","gold"],
                                         labels=["Not_Promoted_Employees","Promoted_Employees"],
                                         shadow=False,
                                         pctdistance=0.8)

#hides the axis,borders and whitespaces.
plt.axis("equal")

#Adds a CENTERED TEXT/TITLE to the figure.
plt.suptitle("Target Class Balance",fontsize=30)

#plt.show() is used to display the figure.
plt.show()

As we can see there is a Huge Data Imbalance between the promoted employees and those who are not, 
So inorder to balance the target classes, we need to balance the target class.
whenever we use a machine learning model there will be a high chance of getting poor results and the results can be biased to the class having higher distributioin.
So inorder to this we can use something called RESAMPLING METHODS.
* __UNDERSAMPLING__:It is the method of eliminating the values from the higher class, so as to bring the higher class to that of smaller one and make it EQUAL.
* __OVERSAMPLING__:Its is a method of bringing the lower class equal to that of upper class, i.e Adding the values so as to make it equal to that of higher class.

report=sweetviz.compare([dodge,"Training_data"],[charger,"Testing_data"],"is_promoted")
report.show_html("my_report.html")

# Checking statistics for the dataset..
Descriptive Statistics..
* for Numerical Columns we check for stats such as Max, Min, Mean, count, standard deviation, 25 percentile, 50 percentile, and 75 percentile.
* for Categorical Columns we check for stats such as count, frequency, top, and unique elements.

#this return the stats for each numerical columns present in the dataset.
dodge.describe()

# this returns the stats for the categorical columns.
#count-the count of the records.
#unique-no of unique elements present in a particular column.
#top-it refers to which unique element/department/any other categorical column is greater apart from other unique elements.
#eg:out of 9 different departments sales&marketing has higher no of records compared to other elements present in department.
#freq-the number of employees present in the list that has toped in a categorical column.
dodge.describe(include="object")

#Inorder to get more detailed view for the stats of numerical columns..
#the “:” Represents to select all rows.
# the integer always signifies the column which we should consider and print. 
# : represents default like select all the rows/columns to print/display.
dodge.iloc[:,1:].describe().style.background_gradient(cmap='copper')

## OUTLIERS
__Outlier can be defined as extremly low or extremly high values in our dataset.__
* it is better to use mean if there is no outlier.
* it is better to use median when there is ouliers in our dataset
* we use mode when we have a categorical variables/values.

## Note
* So our dataset dose'nt consist of any outliers but it consists of missing values that we are going to treat further.
* the average training score for most of the Employee lie between 40 to 100, which is a very good distribution, also the mean is 60.
* Also, the Length of service is not having very disruptive values, so we can keep them for model training, They are'nt going to harm us a lot.

#Now, Lets make an interactive function to check the statistics of these numerical columns at a time.
#in order to use the ipywidgets we need to import something called as @interact before writing any ipywidget function.
#ipywidgtes -> interactive widgets for better understanding and visualisation

@interact
def check(column = list(dodge.select_dtypes('number').columns[0:8])):
    print("Maximum Value :", dodge[column].max())
    print("Minimum Value :", dodge[column].min())
    print("Mean : {0:.2f}".format(dodge[column].mean()))
    print("Median :", dodge[column].median())
    print("Standard Deviation :  {0:.2f}".format(dodge[column].std()))

# dodge.select_dtypes('object').columns[0:8]
dodge.select_dtypes('number').columns[0:8]

# here object refers to categorical columns/non_numerical columns.
dodge.select_dtypes(include = 'object').head()
# dodge.select_dtypes('number').head()

## TREATMENT OF MISSING VALUES
* Treatment of Missing Values is very Important Step in any Machine Learning Model Creation 
* Missing Values can occur due to various reasons, such as the filling incomplete forms, values not available, etc
* There are so many types of Missing Values such as 
     * Missing values at Random
     * Missing values at not Random
     * Missing Values at Completely Random
* What can we do to Impute or Treat Missing values to make a Good Machine Learning Model
    * We can use Business Logic to Impute the Missing Values
    * We can use Statistical Methods such as Mean, Median, and Mode.
    * We can use ML Techniques to impute the Missing values
    * We can delete the Missing values, when the Missing values percentage is very High.
    * if a column has more missing values than the values it actually has, we can completely delete that column in that case or vice versa.
* When to use Mean, and when to use Median?
    * We use Mean, when we do not have Outliers in the dataset for the Numerical Variables.
    * We use Median, when we have outliers in the dataset for the Numerical Variables.
    * We use Mode, When we have Categorical Variables. 

# calculating sum() of all the null values in the dodge file.
dodge_null = dodge.isnull().sum()
# calculating the percentage of null values in dodge(train) data.
# dodge.shape[0] represents the number of rows, shape[1] represents number of columns.
dodge_null_percent = (dodge_null/dodge.shape[0] * 100).round(2)

# calculating sum() of all the null values in the charger(test) file.
charger_null = charger.isnull().sum()
# calculating the percentage of null values in charger(test) data.
# charger.shape[0] represents the number of rows, shape[1] represents number of columns.
charger_null_percent = (charger_null/charger.shape[0] * 100).round(2)

#print(dodge_null,dodge_null_percent,charger_null,charger_null_percent)
# here we merge all the samll df into one by concat([],axis=1,) with keys/ index as ['dodge_null','dodge_null_percent','charger_null','charger_null_percent'].
# axis = 1 refers our result to be in the form of columns, axis = 0 refers to be our result to be in the form of rows.
total_stats = pd.concat([dodge_null,dodge_null_percent,charger_null,charger_null_percent],axis = 1, keys = ['dodge_null','dodge_null_percent','charger_null','charger_null_percent'])
total_stats.style.bar(color = 'magenta')

# here ford hold the column names which consists of null values.
# dtypes - its selects all the columns in the dodge(train) dataset, select_dtype('number'/'object') selects particular type of data we want to display.
# .any() is used to find whether there are any null values in the columns/ not, if there is any null value it will return true or false.
ford = dodge.dtypes[dodge.isnull().any()]
print(ford)

# imputing missing values in columns education and previous_year_rating in dodge(train) data, by using mode()
# fillna() is used to fill the missing values using mode() data.
dodge['education'] = dodge['education'].fillna(dodge['education'].mode()[0])
#dodge['education'].isnull().any()
dodge['previous_year_rating'] = dodge['previous_year_rating'].fillna(dodge['previous_year_rating'].mode()[0])
#dodge['previous_year_rating'].isnull().any()
print("Number of Missing values in dodge(train) data is:", dodge.isnull().sum().sum())

# imputing missing values in columns education and previous_year_rating in charger(test) data, by using mode()
# fillna() is used to fill the missing values using mode() data.
charger['education'] = charger['education'].fillna(charger['education'].mode()[0])
#charger['education'].isnull().any()
charger['previous_year_rating'] = charger['previous_year_rating'].fillna(charger['previous_year_rating'].mode()[0])
#charger['previous_year_rating'].isnull().any()
print("Number of Missing values in charger(test) data is:", charger.isnull().sum().sum())

## Outlier Detection.
Outliers in the dataset are referred as the huge difference between the range of values present in the dataset.
Now, this outliers can result in the poor performance of our predictive model and can result in incorrect results.
there are several methods in order to solve these outliers issues.
* Visualization.


# lets us analyze the numerical columns in dodge(train) dataset.
dodge.select_dtypes('number').head()

## **Exploratory Data Analysis**

## Outlier Detection Using BOX PLOTS On Numerical and Categorical Columns.

# let us analyze each column and see whether it contains ouliers or not.
# lets check the boxplots for the columns.
plt.rcParams['figure.figsize'] = (20,10)
plt.style.use('fivethirtyeight')

# Box Plot for dodge['avg_training_score']
plt.subplot(1,1,1)
sns.boxplot(dodge['avg_training_score'],color='red')
plt.xlabel('Average Training Score',fontsize = 20)
plt.ylabel('Range',fontsize = 20)

plt.show()

# Box Plot for dodge['age']
plt.subplot(1,1,1)
sns.boxplot(dodge['age'],color='Indigo')
plt.xlabel('Age',fontsize = 20)
plt.ylabel('Range',fontsize = 20)

plt.show()

# Univariate Analysis.
Univariate Analysis is a simplest form of statistical analysis, which is used to analyse the columns in the dataset.
#### Univariate refers to analysing single columns(variable) at a time.
* Pie Charts -> these are used when we have very few categories in our categorical columns in our dataset.(eg.kpis met, awards won, previous year rating)
* Count Plots -> these are used when we have more number of categories in our categorical columns in our dataset.(eg.age, no_of_trainings, region, department etc)

dodge.select_dtypes('number').head()

## Analysing Categorical Columns Using PIE CHARTS. 

# Now lets us analyse the categorical coulumns present in our dodge(train) dataset.
plt.rcParams['figure.figsize'] = (30,20)
plt.style.use('fivethirtyeight')


# PIE CHART for the coulumn dodge['previous_year_rating'].
#          (no.columns,no.rows,position)
plt.subplot(2,3,1)
# labels refers to name of the each piece of the pie.
labels = ['1','2','3','4','5']
#        dodge['column_name'].count_of_values_in that_column.
sizes = dodge['previous_year_rating'].value_counts()
#        plt.ColorMap.colormap_name(np.linspace(0,start_color_in_magma,num_colors_we want))
# other color maps (viridis,plasma,inferno,cividis,magma)
colors = plt.cm.Wistia(np.linspace(0, 1, 5))
# explode is use when we want to see the category either the lower category/higher category. 
explode = [0,0,0,0,0.1]
plt.pie(sizes, labels=labels, explode = explode, colors = colors, shadow = True, startangle = 90)
plt.title('Rating',fontsize = 25)



# PIE CHART for column dodge['awards_won?'].

plt.subplot(2,3,2)
plt.title("Awards_Won?",fontsize = 25)
sizes = dodge['awards_won?'].value_counts()
labels = ['0','1']
colors =  plt.cm.viridis(np.linspace(0,1,5))
explode = [0,0.1]
plt.pie(sizes, labels = labels, colors =colors , explode = explode,  shadow = True, startangle = 90)



# PIE CHART for column dodge[KPIs_met >80%'].

plt.subplot(2,3,3)
plt.title('KPIs_met > 80%',fontsize = 25)
sizes = dodge['KPIs_met >80%'].value_counts()
labels = ['0','1']
colors = plt.cm.Wistia(np.linspace(0,1,5))
explode = [0,0]
plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)


# PIE CHART for column dodge['is_promoted'].

plt.subplot(2,3,4)
plt.title('Promoted/Not',fontsize = 25)
sizes = dodge['is_promoted'].value_counts()
labels = ['0','1']
colors = plt.cm.viridis(np.linspace(0,1,5))
explode = [0,0.1]
plt.pie(sizes, labels = labels, colors = colors, explode = explode, shadow = True, startangle = 90)

# legend is an area describing the elements of the graph.
plt.legend()
# plt.show() is used to display the charts/visualizations that we have made.
plt.show()

## Results:
* By seeing the above Charts we can come to a conclusion that,
* Most of The employees are getting very Low ratings.
* Very Few Employees have Won the Awards Recently(2%), like it is very low in a company of employees around 50000+.
* Well, KPI'S Met are very negligibly good when compared to other aspects.
* Lastly, Assuming Overall Employee Performance Chart, Not more than 10% of Employees are getting promoted As expected by their performance.

## Analysing Categorical/Numerical Columns Using COUNT PLOTS. 

dodge.head(20)

# lets analyse the categorical columns which consists of more categories.
# count plot for dodge['age'] categorical column.
plt.rcParams['figure.figsize'] = (10,5)
plt.style.use('fivethirtyeight')

plt.subplot(1,1,1)
plt.hist(dodge['age'],color = 'Red', bins = 20)
plt.xlabel('Age',fontsize=12)

plt.title('Distrubution of Employees Age',fontsize = 12)

plt.grid()
plt.show()

Above is the Count Plot for distribution of age among the employees which shows the count of employees ages present in the dodge dataset.
We can see that employees with age group of 30's are more in number when compared to other age groups, this shows that there are some experienced workers or employees in the organisation.

# lets analyse the categorical columns which are divide by more categories.
# count plot for dodge['no_of_training'] categorical column.
plt.rcParams['figure.figsize'] = (15,5)
plt.style.use('fivethirtyeight')

plt.subplot(1,1,1)
sns.countplot(dodge['no_of_trainings'], palette = 'cividis')
plt.title('Trainings Taken',fontsize = 12)
plt.xlabel('Trainings Taken',fontsize=12)

plt.grid()
plt.show()

Count Plot for no_of_trainings Undertaken by the employees in the organisation.
* we can see that more than 40000+(80%) employees have taken minimum of 1 training in their work experience.
* around negligible amount of employees have taken 2 trainings.
* 5% of employees have taken like 3 trainings.
* 3% of employees who have taken trainings more than thirce.

# lets analyse the categorical columns which are divide by more categories.
# count plot for dodge['region'] categorical column.
plt.rcParams['figure.figsize'] = (12,15)
plt.style.use('fivethirtyeight')

plt.subplot(1,1,1)
sns.countplot(y=dodge['region'], palette = 'cividis', orient = 'v')
plt.title('Distrubution of Regions',fontsize = 12)
plt.ylabel('Regions',fontsize=12)

plt.grid()
plt.show()

# lets analyse the categorical columns which are divide by more categories.
# count plot for dodge['age'] categorical column.
plt.rcParams['figure.figsize'] = (10,5)
plt.style.use('fivethirtyeight')

plt.subplot(1,1,1)
sns.countplot(y=dodge['department'], palette = 'cividis', orient = 'v')
plt.title('Distrubution of Employees in Deaprtments',fontsize = 12)
plt.ylabel('Department',fontsize=12)
plt.grid()
plt.show()

Above Count plot shows the distribution of employees among different departments present in the organisation.
* we can see that Sales & Marketing department is ruling the organisation which has got more number of employees working on it, along with Operation, Technology, Analytics so on..

# lets analyse the categorical columns which are divide by more categories.
# count plot for dodge['length_of_service'] categorical column.
plt.rcParams['figure.figsize'] = (10,5)
plt.style.use('fivethirtyeight')

plt.subplot(1,1,1)
sns.countplot(dodge['length_of_service'],color = 'Red')
plt.xlabel('Service Length',fontsize=12)

plt.title('Distribution of Length Of Service',fontsize = 12)

plt.grid()
plt.show()

The Above Count Plot shows the Distrubution of Length_of_service in the Organisation among Emoloyees.
* we can see that there are very less employees who have 15 years of experience of work in the organisation.

# lets analyse the categorical columns which are divide by more categories.
# count plot for dodge['avg_training_score'] categorical column.
plt.rcParams['figure.figsize'] = (10,5)
plt.style.use('fivethirtyeight')

plt.subplot(1,1,1)
sns.countplot(dodge['avg_training_score'],color = 'Red')
plt.xlabel('Training Score',fontsize=12)

plt.title('Distrubution of Training Score',fontsize = 12)

plt.grid()
plt.show()

### Now let us analyze some categorical Columns which are not Numerical Columns. 

# let us analyse the columns dodge['education'],dodge['gender'] using PIE CHARTS.
plt.rcParams['figure.figsize']=(15,5)
plt.style.use('fivethirtyeight')

# PIE CHART for dodge['education'] column.
plt.subplot(1,3,1)
labels = dodge['education'].value_counts().index
sizes = dodge['education'].value_counts()
explode = [0,0,0.1]
colors = plt.cm.Wistia(np.linspace(0,1,3))
plt.pie(sizes, colors=colors, labels=labels, explode=explode, shadow =True, startangle=90 )
plt.title('Education',fontsize=12)


#PIE CHART for dodge['gender'] column.
plt.subplot(1,3,2)
#labels = dodge['gender'].value_counts().index
labels = ['M','F']
sizes = dodge['gender'].value_counts()
explode = [0,0]
colors = plt.cm.Wistia(np.linspace(0,1,3))
plt.pie(sizes, colors=colors, labels=labels, explode=explode, shadow =True, startangle=90 )
plt.title('Gender',fontsize=12)


#PIE CHART for dodge['recruitment_channel'] column
plt.subplot(1,3,3)
labels = dodge['recruitment_channel'].value_counts().index
sizes = dodge['recruitment_channel'].value_counts()
colors = plt.cm.Wistia(np.linspace(0,1,5))
explode = [0,0,0.1]
plt.pie(sizes, colors=colors, explode=explode, labels=labels, shadow=True, startangle=90)
plt.title('Recruitment Channel',fontsize=12)




plt.legend()
plt.show()

### Results:
* Above We have studied how employee promotion has got involved with the aspects such as KPI's,RATING,AWARDS_WON.
* Now, Lets us discuss some more aspects like how Education, Gender, Recruitment Channel Which Involves in Employees Retention.
* The first Chart Purely Shows that There are lot of Employees With Bachelors Background and very less Employees With Below Secondary Background, Which means People Are not showing interest in working after their Below Secondary Education i.e People are more involved into their higher studies than working.
* Gender - As Expected there are more numbers of Males in the Organisation than Females, Which will never leave them on backfoot of getting promotion.
* Recruitment channel - We can see that very few employees are being referred in a company to work because of the connections they make which is very good, but also most of the employees are being pulled due to sourcing(based on skill they possess) and some Other Private Recruitment channels. 

# BIVARIATE ANALYSIS 
* Bivariate Anlaysis is technique of analysing 2 or more columns at a time, Due to this we can make significant findings, to find how does this columns affect our employees promotion status and to find the relationship between both the  variables/columns.
* It also helps us in simple hypothesis association.

#### Types of Bivariate Analysis.

* Categorical vs Categorical
* Numerical vs Categorical
* Numerical vs Numerical
* First, we will perform Categorical vs Categorical Analysis using Grouped Bar Charts with the help of crosstab function.
* Second, we will perform Categorical vs Numerical Analysis using Bar Charts, Box plots, Strip plots, Swarm plots, Boxen plots, etc
* Atlast, we will perform Numerical vs Numerical Analysis using Scatter plots.

## Categorical vs Categorical 

# Categorical vs Categorical Analysis using Grouped Bar Charts with the help of crosstab function.
# lets check the Effect of gender on Promotion.


import warnings
warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize']=(15,3)

# pd.crosstab() function is a very useful and most advanced fuction in the python dataframe,
# which gives the result by comparing 2 or more columns/variables from the dataset.

x = pd.crosstab(dodge['gender'],dodge['is_promoted'])
# or x.plot(kind='bar', stacked = False) / x.plot(kind='bar') /x.plot()
x.div(x.sum(1).astype(float), axis = 0).plot(kind='bar', stacked = False)

plt.title('Effect Of Gender On Promotion')
plt.xlabel(" ")

* Although we have seen female were minority in the organisation, but with this plot completly rules out that distribution, where we can see conclude that females are giving neck to neck competition to males in getting promoted. 

# Effect of Department on Promotion.

plt.rcParams['figure.figsize'] = (12,5)
plt.title('Effect of Department on Promotion',fontsize=12)
x = pd.crosstab(dodge['department'],dodge['is_promoted'])
x.div(x.sum(1).astype(float), axis = 0).plot(kind = 'bar', stacked = False)
plt.xlabel(' ')
plt.xticks(rotation = 20)
plt.show()

* There isn't much difference of getting promoted based on the departments, where every department has different process and almost every department has equal number of people getting promoted.
* Although Technical and procurement employees are topped in the list.

# Effect of Education on Promotion.

plt.rcParams['figure.figsize'] = (12,5)
plt.title("Effect of Education On Promotion", fontsize = 12)
x = pd.crosstab(dodge['education'], dodge['is_promoted'])
x.plot(kind = 'bar', stacked = False)
plt.xlabel('')
plt.xticks(rotation = 15)
plt.show()

* It is clearly shown that Employees Owing Bachelors degree are most likey to get promoted comapritively to other two educations.
* As Expected there aren't many Employees with Below Secondary Degree who are likely in the race for Promotion, Because Most of the employees/People are showing Interests towards further studies rather than working after their Below Secondary Education.
* Employees owning Masters degree are also negligibly high to get promoted, as there aren't many employees with that Master's who have somewhat more advanced knowledge that other two.

## Numerical vs Categorical 

# Effect of length_of_servie on getting Promotion.

plt.rcParams['figure.figsize'] = (13,6)
plt.title('Effect of Service Length of Promotion', fontsize = 12)
x = pd.crosstab(dodge['length_of_service'],dodge['is_promoted'])
x.plot(kind = 'bar',stacked = False )
plt.xlabel('Service_length')
plt.ylabel('Count')
plt.show()



* It is clearly visible that the Company is not giving first Priority to the employees with ample amount of Work Experience they have in the organisation, but mostly they are going for employees with negligible experience to give Promotion.

# Effect of age on Promotion using Bar plot.

plt.rcParams['figure.figsize'] = (12,5)
plt.title('Effect of age on Promotion', fontsize = 12)
x = pd.crosstab(dodge['age'],dodge['is_promoted'])
x.plot(kind='bar', stacked = False)
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# Effect of age on Promotion using BoxenPlot.

plt.rcParams['figure.figsize'] = (15,4)
plt.title('Effect of age on Promotion', fontsize = 12)
sns.boxenplot(dodge['is_promoted'],dodge['age'], palette = 'PuRd')
plt.ylabel('Age')
plt.xlabel('Promoted/Not')
plt.show()


## Numerical vs Numerical 

dodge.head()

# lets plot for no_of_trainings and avg_training_score

plt.rcParams['figure.figsize'] = (12,5)
plt.title('Effect of Training and Avg_Training_Score', fontsize = 12)
sns.boxplot(dodge['no_of_trainings'],dodge['avg_training_score'])
#x = pd.crosstab(dodge['no_of_trainings'],dodge['avg_training_score'])
#x.plot(kind = 'bar', stacked = False)

plt.xlabel('Trainings')
plt.ylabel('Avg_Training_score')
plt.show()

* It is seen that the Avg Training Score is gradually decreasing for the Employees taking More number of Trainings.
* He/She might not be from a relavant domain of the work they are doing, due to that they might be taking more number of trainings to master the skill they need in the work.
* It is recommended that an Employee should not take more than 2-3 trainings, inorder to be in the race for the promotion or else he/she might get demoted or get fired from the workspace due to lack of knowledge on what they are working on.

* Although we have seen that taking more number of trainings might affect your promotion, But for some employees it can also increase their service length like in the above plot.

# lets check for length_of_service and avg_training_Score.

plt.rcParams['figure.figsize'] = (12,5)
plt.title('Effect of Service Length and Avg Training Score')
sns.stripplot(dodge['length_of_service'], dodge['avg_training_score'])
plt.ylabel('Service Length', fontsize = 12)
plt.xlabel('Avg_Training_Score', fontsize = 12)

plt.show()

* The above visualization explains us about how Avg Training Score plays an important role for an employees length of the service in an organisation.
* So, based on the visualization we can tell that negligible number of employees have increased the length of their service with the Avg Training Score ranging between 43 - 87.

## Interactive Function for Visualising the columns(Bivariate).


# Using Boxplots.
# checking for Object types and Numerical Types.

plt.rcParams['figure.figsize'] = (12,5)
@interact_manual
def interactive_function(column1=list(dodge.select_dtypes('object').columns), column2 = list(dodge.select_dtypes('number').columns[1:])):
    sns.boxplot(dodge[column1],dodge[column2])

# Using Boxenplots.
# checking for Object types and Numerical Types.

plt.rcParams['figure.figsize'] = (12,5)
@interact_manual
def interactive_function(column1=list(dodge.select_dtypes('object').columns), column2 = list(dodge.select_dtypes('number').columns[1:])):
    sns.boxenplot(dodge[column1],dodge[column2])

## Interactive function for Numerical and Numerical types using Box,Boxen,strip

# Using Boxplots.
# checking for Numerical types and Numerical Types.

plt.rcParams['figure.figsize'] = (12,5)
@interact_manual
def interactive_function(column1=list(dodge.select_dtypes('number').columns[1:]), column2 = list(dodge.select_dtypes('number').columns[1:])):
    sns.boxplot(dodge[column1],dodge[column2])

# Using Boxenplots.
# checking for Numerical types and Numerical Types.

plt.rcParams['figure.figsize'] = (12,5)
@interact_manual
def interactive_function(column1=list(dodge.select_dtypes('number').columns[1:]), column2 = list(dodge.select_dtypes('number').columns[1:])):
    sns.boxenplot(dodge[column1],dodge[column2])

# Using stripplots.
# checking for Numerical types and Numerical Types.

plt.rcParams['figure.figsize'] = (12,5)
@interact_manual
def interactive_function(column1=list(dodge.select_dtypes('number').columns[1:]), column2 = list(dodge.select_dtypes('number').columns[1:])):
    sns.stripplot(dodge[column1],dodge[column2])

# MultiVariate Anlaysis

* In this instance, a multivariate analysis would be required to understand the relationship of each variable with each other.


* First, we will use the Correlation Heatmap to check the correlation between the Numerical Columns
* Then we will check the ppscore or the Predictive Score to check the correlation between all the columns present in the data.


# lets plot the correlation heat map for all the numerical columns.
# correlation heat map is used to identify the columns which share same data/contains identical data.
# so that we can just remove any one of them from our dataset to make our model work faster and efficient.

# If the value is 1 that means values are highly correlated.
# If the value is 0 that means their is no similarity / disimilarity.
# If the value is -1 that means their is no correlation between the columns.
# dodge.corr() is the correlation of dodge dataset.

plt.rcParams['figure.figsize'] = (10,10)
plt.title('Correlation Heat Map', fontsize = 13)
sns.heatmap(dodge.corr(),annot=True, linewidth = 0.5, cmap = 'Wistia')

plt.show()

* Through the above heat map we can see that their is a correlation between age and length of service of the employees, obviously if an employee with age 50 years would have work experience of around 20 years, but an employee with 25 years would not have that much experience so, he might not be a part of the organisation.
* Not only Age and Length of Service, We also have correlation between Kpi's Met and Previous Year ratings. 

dodge.info()

# Lets check for Dsitrbution and Length of Service W>R>T Awards Won.

plt.rcParams['figure.figsize'] = (13,7)
plt.title('Distrubution of Department and Promotion Over Awards Won', fontsize = 12)
sns.barplot(dodge['department'],dodge['length_of_service'],dodge['awards_won?'])
plt.xlabel('Department', fontsize = 12)
plt.ylabel('Service Length', fontsize = 12)
plt.show()

# Feature Engineering

* Feature Engineering is the Technique of using Domain Knowledge inorder to extract the important features from the dataset using some data mining techniques.
* These features can in turn help us to create/bulid an efficeient Machine Laerning Model.


* Now, There are lot of methods/techniques to perform feature engineering such as.
* People in the industry consider it has a most important step, due to the important information it will provide by the features extracted.
* Before Starting a Project one need to thoroughly understand each and every column present in the dataset, inorder to extract some new features from the old existing features.
* Different ways of extracting features from the dataset.
 * We can remove unnecessary columns from our dataset.
 * we can perform Binning method on Numerical and Categorical columns.
 * we can extract features from date and time features.
 * we can extract features from the categorical columns.
 * we can even perform aggreagtion on two/more columns together using some aggregating functions.

dodge.head()

# 1.Performing Aggregation on the columns.

dodge['sum_metric'] = dodge['awards_won?'] + dodge['KPIs_met >80%'] + dodge['previous_year_rating']
charger['sum_metric'] = charger['awards_won?'] + charger['KPIs_met >80%'] + charger['previous_year_rating']

dodge['total_score'] = dodge['avg_training_score'] * dodge['no_of_trainings']
charger['total_score'] = charger['avg_training_score'] * charger['no_of_trainings']


dodge.head()

# 2.Removing Unnecessary Columns from the dataset.

dodge = dodge.drop(['employee_id', 'region', 'recruitment_channel'], axis=1)
charger = charger.drop(['employee_id', 'region', 'recruitment_channel'], axis=1)

dodge.head()

# Grouping and Filtering
* Grouping is the fundamental concept and very important function in clustering techniques.
* They are used to find the collections of entities and the relationships they have between each other.
* Grouping and filtering is considered as one of the important steps in analysing/investigating the data.
* It helps in taking important decisions using filters and groupby funtions.

# using crosstab() fucntion we find the distribution of kpis met and awards won.

x = pd.crosstab(dodge['KPIs_met >80%'], dodge['awards_won?'])
x.style.background_gradient(cmap='viridis')

# crosstab() for department and promotion.

x = pd.crosstab(dodge['department'], dodge['is_promoted'])
x.style.background_gradient(cmap='viridis')

#crosstab() for how gender affecting the promotion of employees.

x = pd.crosstab(dodge['gender'], dodge['is_promoted'])
x.style.background_gradient(cmap='magma')

# crosstab() how no_of_trainings affect promotion.

x = pd.crosstab(dodge['no_of_trainings'], dodge['is_promoted'])
x.style.background_gradient(cmap='viridis')

#crosstab for how age affecting the employees promotion, majorly we can see that ages ranging[26-40] are more likey to get promoted, maybe due to their work experience.
x = pd.crosstab(dodge['age'], dodge['is_promoted'])
x.style.background_gradient(cmap='viridis')

x = pd.crosstab(dodge['education'],dodge['is_promoted'])
x.style.background_gradient(cmap= 'Wistia')

## Results.
 * we have seen that using crosstab functions, we have get to known how different columns affect the promotion of the employees, making us understand the dataset even more clearly, to make our prediction model efficient.

# checking awards_won by the employees by grouping through Departments.
# selecting multiple columns is done by below method.
# it can also be done by..
# dodge.groupby('department').count().sort_values(by='awards_won?', ascending = False), by it selects all the colums in the dataset, that's why we go for below method.

dodge[['department','awards_won?']].groupby('department').count().sort_values(by = 'awards_won?' ,ascending=False)

dodge[['length_of_service','awards_won?']].groupby('length_of_service').count().sort_values(by = 'awards_won?', ascending = False)

dodge[['no_of_trainings','length_of_service']].groupby('no_of_trainings').mean()

# lets make an interactive console for grouping different columns.

@interact_manual
def group_operation(column1 = list(dodge.select_dtypes('object').columns), column2 = list(dodge.select_dtypes('number').columns)[1:]):
    return (dodge[[column1,column2]].groupby([column1]).count().style.background_gradient(cmap = 'viridis'))


# interactive manual for grouping 2 numerical columns and finding the count()

@interact_manual
def group_operation(column1 = list(dodge.select_dtypes('number').columns), column2 = list(dodge.select_dtypes('number').columns)[1:]):
    return (dodge[[column1,column2]].groupby([column1]).count().style.background_gradient(cmap = 'magma'))

# interactive console for grouping 2 columns and finding count,mean,max,min functions.

@interact_manual
def group_operation(column1 = list(dodge.select_dtypes('object').columns), column2 = list(dodge.select_dtypes('number').columns)[1:]):
    return (dodge[[column1,column2]].groupby([column1]).agg(['count','mean','max','min']).style.background_gradient(cmap = 'plasma'))

# interactive console for grouping 2 numerical columns and finding count,mean,max,min functions.

@interact_manual
def group_operation(column1 = list(dodge.select_dtypes('number').columns), column2 = list(dodge.select_dtypes('number').columns)[1:]):
    return (dodge[[column1,column2]].groupby([column1]).agg(['count','mean','max','min']).style.background_gradient(cmap = 'cividis'))

#Interactive Manual to check promotion status based on employees no of trainings taken. 

@interact
def check(column = 'no_of_trainings', x = 5):
    y = dodge[dodge['no_of_trainings'] > x]
    return y['is_promoted'].value_counts()
check()

### Replacing no_of_trainings, Dealing with biased records and making Interactive Fucntions.

# since their is negligible values of employees taking trainings more than 5, we can just remove those training values and replace it by 5.

dodge['no_of_trainings'].value_counts()

# replacing employees taken more than 5 trainings with only 5 trainings taken.

dodge['no_of_trainings'] = dodge['no_of_trainings'].replace((6,7,8,9,10),(5,5,5,5,5))
dodge['no_of_trainings'].value_counts()

# lets check for any negative promotions/unfair promotions/ are their any employees who got promotion which they should not..
# like no awards won, no kpis met, low average training score, rating = 1.0

dodge[(dodge['awards_won?'] == 0) & (dodge['KPIs_met >80%'] == 0) & (dodge['avg_training_score'] < 60) & (dodge['previous_year_rating'] == 1.0) & (dodge['is_promoted'] == 1)]

# let us remove those 2 records from the dataset using drop function.

print('Before Deleting the Rows:', dodge.shape)
dodge = dodge.drop(dodge[(dodge['awards_won?'] == 0) & (dodge['KPIs_met >80%'] == 0) & (dodge['avg_training_score'] < 60) & (dodge['previous_year_rating'] == 1.0) & (dodge['is_promoted'] == 1)].index)
print('After Deleting the Rows:',dodge.shape)

# Interactive manual to check the promotion status of an employee with length_of_service > 10.

@interact
def check_promotion(x=10):
    y = dodge[dodge['length_of_service'] > x]
    return y['is_promoted'].value_counts()
check_promotion()

# Interactive manual to check the promotion status of an employee with avg_training_score > 50.

@interact
def check_promotion(x=50):
    y = dodge[dodge['avg_training_score'] > x]
    return y['is_promoted'].value_counts()
check_promotion()

# Dealing With Categorical Columns.

* Dealing with categorical columns is a big step in a project, as sometimes they might hold the important values which help our model to predict accurately.
* Changing our categorical columns into numerical type is a very important step as most of the machine learning models works using numerical values compared to categorical/object type.
* There are several ways of converting a categorical object type into numerical type such as.
    * Here, we are going to use Business Logic to encode the education column / traditional replace() method.
    * Then we will use the Label Encoder, to Department and Gender Columns, which is part of machine learning module(sklearn).

dodge.select_dtypes('object').head()

dodge.head()

# lets convert the education column to numerical values by replacing(Masters & Above -> 3, Bachelor's -> 2, Below Secondary -> 1)

dodge['education'] = dodge['education'].replace(("Master's & above", "Bachelor's", 'Below Secondary'),(3,2,1)) 
charger['education'] = charger['education'].replace(("Master's & above", "Bachelor's", 'Below Secondary'),(3,2,1))

### LabelEncoder for encoding string into numbers using sklearn.preprocessing.
* Lets convert department and gender into numerical using label encoding technique using module sklearn.preprocessor import LabelEncoder.

* LabelEncoder -> Encode target labels with value between 0 and n_classes-1.
* it will first identify the strings in the columns and sorts them in ascending order then assigns the string with a number from (0 to nclasses-1).

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

dodge['department'] = le.fit_transform(dodge['department'])
charger['department'] = le.fit_transform(charger['department'])

dodge['gender'] = le.fit_transform(dodge['gender'])
charger['gender'] = le.fit_transform(charger['gender'])

dodge.head()

# Splitting of Data.
* Splitting of data is often considered as one of the important stpes of building a machine model, where we need to seperate/ split our target_column into separate column form the original datset.
* In our case we just drop our target_column [is_promoted] from the dataset and store it in a seperate dataframe, and rest of the columns in another dataframe.
* Also, we change our test dataset name from charger to test_data.

lexus = dodge['is_promoted']
mustang = dodge.drop(['is_promoted'],axis=1)
test_data = charger

print("shape of Lexus is: ", lexus.shape)
print("shape of mustang(dodge) is: ", mustang.shape)
print("shape of test_data(charger) is: ", test_data.shape)

lexus.value_counts()

lexus.shape

# Resampling
* Resampling is the method that is used to Balance our dataset.


* This Resampling comes to handy whenever our variables/columns are highly imbalanced, so we need to balance them by using some of these resmapling methods, in our case we have seen that our target variable is highly imbalanced.
* There are many Statistical Methods we can use for Resampling the Data such as:
    * Over Samping
    * Cluster based Sampling
    * Under Sampling.
* Random oversampling involves randomly selecting examples from the minority class, with replacement, and adding them to the training dataset.
* Random undersampling involves randomly selecting examples from the majority class and deleting them from the training dataset."In the random under-sampling, the majority class instances are discarded at random until a more balanced distribution is reached."
* In cluster sampling, researchers divide a population into smaller groups known as clusters.  They then randomly select among these clusters to form a sample.
    

# for our dataset, we are going to use oversampling method instead of undersampling as we do not want to lose any of our data in our target_variable.
# lets import the SMOTE algorithm which does same stuff as oversampling does.
# SMOTE -> SMOTE (synthetic minority oversampling technique) is one of the most commonly used oversampling methods to solve the imbalance problem. 
# It aims to balance class distribution by randomly increasing minority class examples by replicating them.

from imblearn.over_sampling import SMOTE
x_resample, y_resample = SMOTE().fit_sample(mustang,lexus.values.ravel())
print(x_resample.shape)
print(y_resample.shape)

# lets compare our results before and after doing resampling.

print("Before Resampling Target_Variable: ")
print(lexus.value_counts())

y_resample = pd.DataFrame(y_resample)
print("After Resampling Target_Variable:")
print(y_resample[0].value_counts())

# Lets Create Validation sets for the training data, so that we can check whether the model that we have created is good enough or not.
# lets import train_test_split module from sklearn package.
# Training Dataset -> The actual dataset that we use to train the model (weights and biases in the case of a Neural Network). The model sees and learns from this data.
# Validation set ->  The sample of data used to provide an unbiased evaluation of a model fit on the training dataset while tuning model hyperparameters. 
# Test Dataset ->  The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.
# it is used only when the model is completly trainied.

from sklearn.model_selection import train_test_split

# train_test_split is a technique of splitting the datasets in x_train,x_test/x_valid,y_train,y_test_y_valid.
# here our datasets(x_resample,y_resample) is splitted in 2 parts by test_size = 0.2/20% i.e (x_train = 80%, x_test/x_valid = 20%)
# 
x_train,x_valid,y_train,y_valid = train_test_split(x_resample,y_resample, test_size = 0.2, random_state=0)

# lets print the shapes of our validation sets.
print('Shape of x_train(mustang/dodge): ', x_train.shape)
print('Shape of y_train(lexus/is_promoted):',y_train.shape)
print('Shape of x_valid(mustang/dodge): ', x_valid.shape)
print('Shape of y_valid(lexus/is_promoted): ', y_valid.shape)
print('Shape of test_data(charger):',test_data.shape)


# Feature Scaling 
* Feature scaling is the method of scaling the features in our dataset, so that the higher values dosent dominate the lower values.
* Scaling can make a difference between a weak machine learning model and a better one.
    
* Why do we need scaling?
* Machine learning algorithm just sees number — if there is a vast difference in the range say few ranging in thousands and few ranging in the tens, and it makes the underlying assumption that higher ranging numbers have superiority of some sort.
    
    
* The most common techniques of feature scaling are Normalization and Standardization.
* Normalization is used when we want to bound our values between two numbers, typically, between [0,1] or [-1,1].
* Standardization transforms the data to have zero mean and a variance of 1, they make our data unitless.





# It is very important step to do scaling for all our features in our dataset, in order to bring them on to same scale.
# In our case we use standardization to do scaling for our dataset.

# lets import standization scale from the sklearn M.L package.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

# lets perfrom standardization for our x_train,x_valid datasets.

x_train = sc.fit_transform(x_train)
x_valid = sc.transform(x_valid)
test_data = sc.transform(test_data)


# Machine Learning Predictive Modelling.
* Predictive modelling is the mechanism of predicting the outcomes of the datasets, by using the data and statistics.
* These models can be used to predict anything from sports outcomes and TV ratings to technological advances and corporate earnings. 
* Predictive modeling is also often referred to as: Predictive analytics.

# Decision Tree Classifier.
* A decision tree is a flowchart-like tree structure where an internal node represents feature(or attribute), the branch represents a decision rule, and each leaf node represents the outcome.
* The basic idea behind any decision tree algorithm is as follows:
    * Select the best attribute using Attribute Selection Measures(ASM) to split the records.
    * Make that attribute a decision node and breaks the dataset into smaller subsets.
    * Starts tree building by repeating this process recursively for each child until one of the condition will match:
        
        
   * Attribute selection is done by using either of this methods:
       * Information Gain
       * Gain Ratio
       * Gini index
       
       
       


dodge.head()

# lets import decision tree clasifier from the sklearn ML package.
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report

model = DecisionTreeClassifier()
# lets train our model with DecisionTreeClassifier()
model.fit(x_train,y_train)
# lets predict the outcome of the data(x_valid)
y_predict = model.predict(x_valid)


print('Training Accuracy is: ', model.score(x_train,y_train))
print('Testing Accuracy is: ', model.score(x_valid,y_valid))

# anything betweeen 65-75 % is termed as a bad accuracy rate.
# Note: If there is a large difference between Training and Testing Accuracy then it can be called as overfitting.


# lets us make a confusion matrix to understand the actual_target_values with the predicted_values(y_predict).
# Confusion Matrix helps us to analyse the mistakes which are done by our machine learning model.

cm = confusion_matrix(y_valid,y_predict)
plt.rcParams['figure.figsize'] = (5,5)
plt.title("Confusion Matrix for Actual and Predicted", fontsize = 13)
sns.heatmap(cm, annot = True, cmap = 'Wistia', fmt = '.8g')
plt.ylabel('Actual', fontsize = 13)
plt.xlabel('Predicted', fontsize = 13)
plt.show()

# Feature Selection for Decision Tree Model.
* Feature selection for Decision Tree Model helps us to select which features are helping our model to get good predictions and which are not.


import warnings
warnings.filterwarnings('ignore')

# lets import feature_selection from sklearn package.
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct classifications
# RFECV -> Recursive Feature Elimination Cross Validation, which checks each column in our dataset and validates whether that column is contributing to a good prediction outcome,
# if it is not, then that particular column is removed form the dataset.

model = DecisionTreeClassifier() 
rfecv = RFECV(estimator = model, step = 1, cv = 5, scoring = 'accuracy')
rfecvv = rfecv.fit(x_train, y_train)

print('Optimal number of features :', rfecvv.n_features_)
# the optimal features always varies whenever we execute our cell, so this might impact our predictions too.
x_trainn = pd.DataFrame(x_train)
print('Best features :', x_trainn.columns[rfecvv.support_])

# Classification_Report
cr = classification_report(y_valid,y_predict)
print(cr)

## Lets check the Descriptive Stats for all the columns and Perform Some Real TIme Predictions.

dodge.describe()

# lets perform some Real time predictions on top of the Model that we just created using Decision Tree Classifier.

# lets check the parameters we have in our Model
'''

department            -> The values are from 0 to 8, (Department does not matter a lot for promotion)
education             -> The values are from 0 to 3 where Masters-> 3, Btech -> 2, and secondary ed -> 1
gender                -> the values are 0 for female, and 1 for male
no_of_trainings       -> the values are from 0 to 5
age                   -> the values are from 20 to 60
previou_year_rating   -> The values are from 1 to 5
length_of service     -> The values are from 1 to 37
KPIs_met >80%         -> 0 for Not Met and 1 for Met
awards_won>           -> 0-no, and 1-yes
avg_training_score    -> ranges from 40 to 99
sum_metric            -> ranges from 1 to 7
total_score           -> 40 to 710

'''

# Note: the prediction varies whenever we execute the block of cells.

predictions = rfecvv.predict(np.array([[2, #department code
                                      3, #masters degree
                                      1, #male
                                      1, #1 training
                                      30, #30 years old
                                      5, #previous year rating
                                      10, #length of service
                                      1, #KPIs met >80%
                                      1, #awards won
                                      99, #avg training score
                                      7, #sum of metric 
                                      700 #total score
                                     ]]))
print("Whether the Employee should get a Promotion : 1-> Promotion, and 0-> No Promotion :", predictions)
    





## **Testing on Test Data.**

test_predictions = rfecvv.predict(test_data)

predict_df = pd.DataFrame(test_predictions)

predict_df

### **Concat test_data with the predictions.**

queen = pd.concat([charger,predict_df], axis = 1)

queen.rename(columns={0:'is_promoted'}, inplace = True)

queen.head()

queen[queen['is_promoted'] == 1]

