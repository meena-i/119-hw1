# Collaborated with Nisha Thiagaraj and Ruba Thekkath

"""
Part 1: Data Processing in Pandas

**Released: Monday, October 14**

=== Instructions ===

There are 22 questions in this part.
For each part you will implement a function (q1, q2, etc.)
Each function will take as input a data frame
or a list of data frames and return the answer
to the given question.

To run your code, you can run `python3 part1.py`.
This will run all the questions that you have implemented so far.
It will also save the answers to part1-answers.txt.

=== Dataset ===

In this part, we will use a dataset of world university rankings
called the "QS University Rankings".

The ranking data was taken 2019--2021 from the following website:
https://www.topuniversities.com/university-rankings/world-university-rankings/2021

=== Grading notes ===

- Once you have completed this part, make sure that
  your code runs, that part1-answers.txt is being re-generated
  every time the code is run, and that the answers look
  correct to you.

- Be careful about output types. For example if the question asks
  for a list of DataFrames, don't return a numpy array or a single
  DataFrame. When in doubt, ask on Piazza!

- Make sure that you remove any NotImplementedError exceptions;
  you won't get credit for any part that raises this exception
  (but you will still get credit for future parts that do not raise it
  if they don't depend on the previous parts).

- Make sure that you fill in answers for the parts
  marked "ANSWER ___ BELOW" and that you don't modify
  the lines above and below the answer space.

- Q6 has a few unit tests to help you check your work.
  Make sure that you removed the `@pytest.mark.skip` decorators
  and that all tests pass (show up in green, no red text!)
  when you run `pytest part3.py`.

- For plots: There are no specific requirements on which
  plotting methods you use; if not specified, use whichever
  plot you think might be most appropriate for the data
  at hand.
  Please ensure your plots are labeled and human-readable.
  For example, call .legend() on the plot before saving it!

===== Questions 1-6: Getting Started =====

To begin, let's load the Pandas library.
"""

import pandas as pd

"""
1. Load the dataset into Pandas

Our first step is to load the data into a Pandas DataFrame.
We will also change the column names
to lowercase and reorder to get only the columns we are interested in.

Implement the rest of the function load_input()
by filling in the parts marked TODO below.

Return as your answer to q1 the number of dataframes loaded.
(This part is implemented for you.)
"""

NEW_COLUMNS = ['rank', 'university', 'region', 'academic reputation', 'employer reputation', 'faculty student', 'citations per faculty', 'overall score']

def load_input():
    """
    Input: None
    Return: a list of 3 dataframes, one for each year.
    """

    # Load the input files and return them as a list of 3 dataframes.
    df_2019 = pd.read_csv('data/2019.csv', encoding='latin-1')
    df_2020 = pd.read_csv('data/2020.csv', encoding='latin-1')
    df_2021 = pd.read_csv('data/2021.csv', encoding='latin-1')

    # Standardizing the column names
    df_2019.columns = df_2019.columns.str.lower()
    df_2020.columns = df_2019.columns.str.lower()
    df_2021.columns = df_2019.columns.str.lower()

    # Restructuring the column indexes
    # Fill out this part. You can use column access to get only the
    # columns we are interested in using the NEW_COLUMNS variable above.
    # Make sure you return the columns in the new order.
    # TODO
    df_2019 = df_2019[NEW_COLUMNS]
    df_2020 = df_2020[NEW_COLUMNS]
    df_2021 = df_2021[NEW_COLUMNS]

    # When you are done, remove the next line...
    #raise NotImplementedError

    # ...and keep this line to return the dataframes.
    return [df_2019, df_2020, df_2021]

def q1(dfs):
    # As the "answer" to this part, let's just return the number of dataframes.
    # Check that your answer shows up in part1-answers.txt.
    return len(dfs)

"""
2. Input validation

Let's do some basic sanity checks on the data for Q1.

Check that all three data frames have the same shape,
and the correct number of rows and columns in the correct order.

As your answer to q2, return True if all validation checks pass,
and False otherwise.
"""

def q2(dfs):
    """
    Input: Assume the input is provided by load_input()

    Return: True if all validation checks pass, False otherwise.

    Make sure you return a Boolean!
    From this part onward, we will not provide the return
    statement for you.
    You can check that the "answers" to each part look
    correct by inspecting the file part1-answers.txt.
    """
    # Check:
    # - that all three dataframes have the same shape
    shape = dfs[0].shape
    rows = len(dfs[0])
    cols = len(NEW_COLUMNS)

    for df in dfs:
        if df.shape != shape:
            return False
    # - the number of rows
        if len(df) != rows:
            return False
    # - the number of columns
        if len(df.columns) != cols:
            return False
    # - the columns are listed in the correct order
        if list(df.columns) != NEW_COLUMNS:
            return False
    return True
    #raise NotImplementedError

"""
===== Interlude: Checking your output so far =====

Run your code with `python3 part1.py` and open up the file
output/part1-answers.txt to see if the output looks correct so far!

You should check your answers in this file after each part.

You are welcome to also print out stuff to the console
in each question if you find it helpful.
"""

ANSWER_FILE = "output/part1-answers.txt"

def interlude():
    print("Answers so far:")
    with open(f"{ANSWER_FILE}") as fh:
        print(fh.read())

"""
===== End of interlude =====

3a. Input validation, continued

Now write a validate another property: that the set of university names
in each year is the same.
As in part 2, return a Boolean value.
(True if they are the same, and False otherwise)

Once you implement and run your code,
remember to check the output in part1-answers.txt.
(True if the checks pass, and False otherwise)
"""

def q3(dfs):
    # Check:
    # - that the set of university names in each year is the same
    # Return:
    # - True if they are the same, and False otherwise.
    names = set(dfs[0]['university'])
    for df in dfs:
        if set(df['university']) != names:
            return False
    return True
    #raise NotImplementedError

"""
3b (commentary).
Did the checks pass or fail?
Comment below and explain why.

=== ANSWER Q3b BELOW ===
The checks failed because even though each dataframe has the same number of schools,
they do not have all of the same schools listed.
=== END OF Q3b ANSWER ===
"""

"""
4. Random sampling

Now that we have the input data validated, let's get a feel for
the dataset we are working with by taking a random sample of 5 rows
at a time.

Implement q4() below to sample 5 points from each year's data.

As your answer to this part, return the *university name*
of the 5 samples *for 2021 only* as a list.
(That is: ["University 1", "University 2", "University 3", "University 4", "University 5"])

Code design: You can use a for for loop to iterate over the dataframes.
If df is a DataFrame, df.sample(5) returns a random sample of 5 rows.

Hint:
    to get the university name:
    try .iloc on the sample, then ["university"].
"""

def q4(dfs):
    # Sample 5 rows from each dataframe
    # Print out the samples
    spl = []
    for df in dfs:
        spl = df.sample(5)
        print(spl)
        
    #raise NotImplementedError

    # Answer as a list of 5 university names
    return spl['university'].tolist()

"""
Once you have implemented this part,
you can run a few times to see different samples.

4b (commentary).
Based on the data, write down at least 2 strengths
and 3 weaknesses of this dataset.

=== ANSWER Q4b BELOW ===
Strengths:
1. The scores are all in percentages, which makes it consistent among all schools 
and easy to compare.
2. There are schools from all over the world, meaning there is a large variety 
and representation in the dataset.

Weaknesses:
1. It doesn't say what the overall score column is based on. I assumed it was the 
average off all the other columns, but that doesn't add up.
2. It might not take into account the differences in academic systems across the 
world when rating them.
3. The rankings probably include all fields of study at the school, so a school 
that focuses its resources towards a specific field (like engineering or art) 
might have a misleading rating.
=== END OF Q4b ANSWER ===
"""

"""
5. Data cleaning

Let's see where we stand in terms of null values.
We can do this in two different ways.

a. Use .info() to see the number of non-null values in each column
displayed in the console.

b. Write a version using .count() to return the number of
non-null values in each column as a dictionary.

In both 5a and 5b: return as your answer
*for the 2021 data only*
as a list of the number of non-null values in each column.

Example: if there are 5 null values in the first column, 3 in the second, 
4 in the third, and so on, you would return
    [5, 3, 4, ...]
"""

def q5a(dfs):
    # TODO
    print(dfs[2].info())

    #raise NotImplementedError
    # Remember to return the list here
    # (Since .info() does not return any values,
    # for this part, you will need to copy and paste
    # the output as a hardcoded list.)
    return [100, 100, 100, 100, 100, 100, 100, 100]

def q5b(dfs):
    # TODO
    counts = dfs[2].count()
    counts_dict = counts.to_list()

    #raise NotImplementedError
    # Remember to return the list here
    return counts_dict
"""
5c.
One other thing:
Also fill this in with how many non-null values are expected.
We will use this in the unit tests below.
"""

def q5c():
    #raise NotImplementedError
    # TODO: fill this in with the expected number
    num_non_null = 100
    return num_non_null

"""
===== Interlude again: Unit tests =====

Unit tests

Now that we have completed a few parts,
let's talk about unit tests.
We won't be using them for the entire assignment
(in fact we won't be using them after this),
but they can be a good way to ensure your work is correct
without having to manually inspect the output.

We need to import pytest first.
"""

import pytest

"""
The following are a few unit tests for Q1-5.

To run the unit tests,
first, remove (or comment out) the `@pytest.mark.skip` decorator
from each unit test (function beginning with `test_`).
Then, run `pytest part1.py` in the terminal.
"""

#@pytest.mark.skip
def test_q1():
    dfs = load_input()
    assert len(dfs) == 3
    assert all([isinstance(df, pd.DataFrame) for df in dfs])

#@pytest.mark.skip
def test_q2():
    dfs = load_input()
    assert q2(dfs)

#@pytest.mark.skip
@pytest.mark.xfail
def test_q3():
    dfs = load_input()
    assert q3(dfs)

#@pytest.mark.skip
def test_q4():
    dfs = load_input()
    samples = q4(dfs)
    assert len(samples) == 5

#@pytest.mark.skip
def test_q5():
    dfs = load_input()
    answers = q5a(dfs) + q5b(dfs)
    assert len(answers) > 0
    num_non_null = q5c()
    for x in answers:
        assert x == num_non_null

"""
6a. Are there any tests which fail?

=== ANSWER Q6a BELOW ===
Yes, the test for q3 failed.
=== END OF Q6a ANSWER ===

6b. For each test that fails, is it because your code
is wrong or because the test is wrong?

=== ANSWER Q6b BELOW ===
I think the test failed because the function returns False as its value when the test was expecting True.
=== END OF Q6b ANSWER ===

IMPORTANT: for any failing tests, if you think you have
not made any mistakes, mark it with
@pytest.mark.xfail
above the test to indicate that the test is expected to fail.
Run pytest part1.py again to see the new output.

6c. Return the number of tests that failed, even after your
code was fixed as the answer to this part.
(As an integer)
Please include expected failures (@pytest.mark.xfail).
(If you have no failing or xfail tests, return 0.)
"""

def q6c():
    # TODO
    return 1
    #raise NotImplementedError

"""
===== End of interlude =====

===== Questions 7-10: Data Processing =====

7. Adding columns

Notice that there is no 'year' column in any of the dataframe. As your first task, append an appropriate 'year' column in each dataframe.

Append a column 'year' in each dataframe. It must correspond to the year for which the data is represented.

As your answer to this part, return the number of columns in each dataframe after the addition.
"""

def q7(dfs):
    # TODO appending year columns to the dataframes and returning the number of cols in each df
    year = 2019
    num_cols = []
    for df in dfs:
        df['year'] = year
        num_cols.append(df.shape[1])
        year+=1
        
    #raise NotImplementedError
    # Remember to return the list here
    return num_cols
"""
8a.
Next, find the count of universities in each region that made it to the Top 100 each year. Print all of them.

As your answer, return the count for "USA" in 2021.
"""

def q8a(dfs):
    # count of universities in top 100 by region
    # TODO
    for df in dfs:
        print(df.groupby('region').size())
    #raise NotImplementedError
    # Remember to return the count here
    return dfs[2].groupby(['region']).size()['USA']

"""
8b.
Do you notice some trend? Comment on what you observe and why might that be consistent throughout the years.

=== ANSWER Q8b BELOW ===
I noticed that USA had the most universities in the Top 100 list, UK had second
most, and Canada had third for all three years. Those were the only three regions
with more than 10 universities on the list, and all the other regions had single
digits. This might be the case because of more funding in these regions and better
resources avaiable at the universities.
=== END OF Q8b ANSWER ===
"""

"""
9.
From the data of 2021, find the average score of all attributes for all universities.

As your answer, return the list of averages (for all attributes)
in the order they appear in the dataframe:
academic reputation, employer reputation, faculty student, citations per faculty, overall score.

The list should contain 5 elements.
"""

def q9(dfs):
    # averages of each column in a list
    # TODO
    avgs = dfs[2].describe().loc['mean'].iloc[1:6].tolist()
    #raise NotImplementedError
    # Return the list here
    return avgs

"""
10.
From the same data of 2021, now find the average of *each* region for **all** attributes **excluding** 'rank' and 'year'.

In the function q10_helper, store the results in a variable named **avg_2021**
and return it.

Then in q10, print the first 5 rows of the avg_2021 dataframe.
"""

def q10_helper(dfs):
    # average of each col by region
    # TODO
    avg_2021 = dfs[2].groupby(['region'])[['academic reputation', 'employer reputation', 
                                'faculty student', 'citations per faculty', 'overall score']].mean()
    return avg_2021

def q10(avg_2021):
    """
    Input: the avg_2021 dataframe
    Print: the first 5 rows of the dataframe

    As your answer, simply return the number of rows printed.
    (That is, return the integer 5)
    """
    # Enter code here
    print(avg_2021.head())
    #raise NotImplementedError
    return 5
"""
===== Questions 11-14: Exploring the avg_2021 dataframe =====

11.
Sort the avg_2021 dataframe from the previous question based on overall score in a descending fashion (top to bottom).

As your answer to this part, return the first row of the sorted dataframe.
"""
# returns row with highest overall score
def q11(avg_2021):
    #raise NotImplementedError
    sorted_avg_2021 = avg_2021.sort_values(by='overall score', ascending = False)
    print(sorted_avg_2021)
    return sorted_avg_2021.iloc[0]

"""
12a.
What do you observe from the table above? Which country tops the ranking?

What is one country that went down in the rankings?
(You will need to load the data and get the 2020 data to answer this part.
You may choose to do this
by writing another function like q10_helper and running q11,
or you may just do it separately
(e.g., in a Python shell) and return the name of the university that you found went
down in the rankings.)

For the answer to this part return the name of the country that tops the ranking and the name of one country that went down in the rankings.
"""

def q12a(avg_2021):
    #raise NotImplementedError
    #Merged df i used to find which countries went down in ranking:
    #merged_df = pd.merge(dfs[0].groupby(['region'])[[ 'overall score']].mean().sort_values(by='overall score', ascending = False)[['overall score']], 
    #avg_2021.sort_values(by='overall score', ascending = False)[['overall score']], on='region')
    return ('Singapore', "Canada")

"""
12b.
Comment on why the country above is at the top of the list.
(Note: This is an open-ended question.)

=== ANSWER Q12b BELOW ===
Singapore is at the top of the list for their academic reputation, employer reputaion, faculty
student ratio, and citations per faculty rankings. This must mean that they have an effective 
academic system and teaching styles. They might also have smaller class sizes, which means the 
students can get personalized help a lot easier than at larger universities.
=== END OF Q12b ANSWER ===
"""

"""
13a.
Represent all the attributes in the avg_2021 dataframe using a box and whisker plot.
Store your plot in output/13a.png.
As the answer to this part, return the name of the plot you saved.
**Hint:** You can do this using subplots (and also otherwise)
"""

import matplotlib.pyplot as plt

def q13a(avg_2021):
    # Plot the box and whisker plot
    # TODO
    plt.figure(figsize=(12, 8))

    boxplot = avg_2021.boxplot()

    plt.title('Box and Whisker Plot of Attributes for avg_2021')
    plt.xlabel('Attributes')
    plt.ylabel('Values')
    plt.grid(visible=False)

    plt.savefig('output/13a.png')
    plt.close()
    #raise NotImplementedError
    return "output/13a.png"

"""
b. Do you observe any anomalies in the box and whisker
plot?

=== ANSWER Q13b BELOW ===
There is an outlier for overall score on the higher end, but I don't really see any anomalies
besides that. Academic reuptation has a very large range, and so do employer reputation and
faculty student ratio.
=== END OF Q13b ANSWER ===
"""

"""
14a.
Pick two attributes in the avg_2021 dataframe
and represent them using a scatter plot.

Store your plot in output/14a.png.

As the answer to this part, return the name of the plot you saved.
"""

def q14a(avg_2021):
    # Enter code here
    # TODO
    plt.figure(figsize=(10, 6))

    plt.scatter(avg_2021['academic reputation'], avg_2021['overall score'])

    plt.title('Scatter Plot of Academic Reputation vs Overall Score')
    plt.xlabel("Academic Reputation")
    plt.ylabel("Overall Score")

    plt.savefig('output/14a.png')
    plt.close()
    #raise NotImplementedError
    return "output/14a.png"

"""
Do you observe any general trend?

=== ANSWER Q14b BELOW ===
There seems to be a positive trend between academic reputation and overall score. It looks
somewhat quadratic, but the correlation is not very strong.
=== END OF Q14b ANSWER ===

===== Questions 15-20: Exploring the data further =====

We're more than halfway through!

Let's come to the Top 10 Universities and observe how they performed over the years.

15. Create a smaller dataframe which has the top ten universities from each year, 
and only their overall scores across the three years.

Hint:

*   There will be four columns in the dataframe you make
*   The top ten universities are same across the three years. Only their rankings differ.
*   Use the merge function. You can read more about how to use it in the documentation: 
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.merge.html
*   Shape of the resultant dataframe should be (10, 4)

As your answer, return the shape of the new dataframe.
"""

def q15_helper(dfs):
    # Return the new dataframe
    # TODO
    top_10 = dfs[0][['university','overall score']].merge(dfs[1][['university', 'overall score']], 
                                                on='university').merge(dfs[2][['university', 'overall score']], 
                                                                       on='university').nlargest(10, 'overall score')
    return top_10

def q15(top_10):
    # Enter code here
    # TODO
    return top_10.shape
    #raise NotImplementedError

"""
16.
You should have noticed that when you merged,
Pandas auto-assigned the column names. Let's change them.

For the columns representing scores, rename them such that they describe the data that the column holds.

You should be able to modify the column names directly in the dataframe.
As your answer, return the new column names.
"""
# changing column names to show which year the overall score corresponds to
def q16(top_10):
    # Enter code here
    # TODO
    #raise NotImplementedError
    top_10.columns = ['university', 'overall score 2019', 'overall score 2020', 'overall score 2021']
    return top_10.columns.to_list()
"""
17a.
Draw a suitable plot to show how the overall scores of the Top 10 universities varied over 
the three years. Clearly label your graph and attach a legend. Explain why you chose the 
particular plot.

Save your plot in output/16.png.

As the answer to this part, return the name of the plot you saved.

Note:
*   All universities must be in the same plot.
*   Your graph should be clear and legend should be placed suitably
"""

def q17a(top_10):
    # Enter code here
    # TODO
    plt.figure(figsize=(12, 8))

    #iterate over each year to plot overall scores for each university over time
    for index, row in top_10.iterrows():
        plt.plot(['2019', '2020', '2021'], 
                 [row['overall score 2019'], row['overall score 2020'], row['overall score 2021']],
                 marker='o', label=row['university'])
    
    plt.title('Change in University Scores Over Time (2019-2021)')
    plt.xlabel('Year')
    plt.ylabel('Score')

    plt.legend(title="University", bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout() #so the legend doesn't overlap on the graph
    plt.savefig('output/17a.png')
    plt.close()
    #raise NotImplementedError
    return "output/17a.png"

"""
17b.
What do you observe from the plot above? Which university has remained consistent in their scores? Which have increased/decreased over the years?

=== ANSWER Q17a BELOW ===
All of the universities remained consistent from 2020 to 2021. However, from 2019 to 2020,
MIT and Stanford were the only ones that remained consistent. Harvard, Caltech, University
of Cambridge, and University of Chicago all decreased. University of Oxford, ETH Zurich, UCL,
and Imperial College London all increased.
=== END OF Q17b ANSWER ===
"""

"""
===== Questions 18-19: Correlation matrices =====

We're almost done!

Let's look at another useful tool to get an idea about how different variables are corelated 
to each other. We call it a **correlation matrix**

A correlation matrix provides a correlation coefficient (a number between -1 and 1) that tells how 
strongly two variables are correlated. Values closer to -1 mean strong negative correlation whereas 
values closer to 1 mean strong positve correlation. Values closer to 0 show variables having no or 
little correlation.

You can learn more about correlation matrices from here: 
https://www.statology.org/how-to-read-a-correlation-matrix/

18.
Plot a correlation matrix to see how each variable is correlated to another. 
You can use the data from 2021.

Print your correlation matrix and save it in output/18.png.

As the answer to this part, return the name of the plot you saved.

**Helpful link:** https://datatofish.com/correlation-matrix-pandas/
"""
import seaborn as sn
def q18(dfs):
    # Enter code here
    # TODO
    corr_mat = dfs[2].iloc[:, 3:8].corr()

    plt.figure(figsize=(12, 8))
    sn.heatmap(corr_mat, annot=True)
    plt.tight_layout()
    plt.show()
    plt.savefig('output/18.png')
    plt.close()
    #raise NotImplementedError
    return "output/18.png"

"""
19. Comment on at least one entry in the matrix you obtained in the previous
part that you found surprising or interesting.

=== ANSWER Q19 BELOW ===
I expected academic reputation to have a large correlation with overall
score, but it only has a value of 0.39.
=== END OF Q19 ANSWER ===
"""

"""
===== Questions 20-23: Data manipulation and falsification =====

This is the last section.

20. Exploring data manipulation and falsification

For fun, this part will ask you to come up with a way to alter the
rankings such that your university of choice comes out in 1st place.

The data does not contain UC Davis, so let's pick a different university.
UC Berkeley is a public university nearby and in the same university system,
so let's pick that one.

We will write two functions.
a.
First, write a function that calculates a new column
(that is you should define and insert a new column to the dataframe whose value
depends on the other columns)
and calculates
it in such a way that Berkeley will come out on top in the 2021 rankings.

Note: you can "cheat"; it's OK if your scoring function is picked in some way
that obviously makes Berkeley come on top.
As an extra challenge to make it more interesting, you can try to come up with
a scoring function that is subtle!

b.
Use your new column to sort the data by the new values and return the top 10 universities.
"""

def q20a(dfs):
    # TODO
    # providing weights for each column that make it so that berkeley is #1
    dfs[2]['new_score'] = (
        0.0 * dfs[2]['academic reputation'] - 
        0.05 * dfs[2]['employer reputation'] +  
        0.05 * dfs[2]['faculty student'] +      
        0.9 * dfs[2]['citations per faculty']   
    )
    
    #raise NotImplementedError
    # For your answer, return the score for Berkeley in the new column.
    berkeley_score = dfs[2][dfs[2]['university'] == 'University of California, Berkeley (UCB)']['new_score'].values[0]
    return berkeley_score

def q20b(dfs):
    # TODO
    df_2021_sorted = dfs[2].sort_values(by='new_score', ascending=False)
    #raise NotImplementedError

    # For your answer, return the top 10 universities.
    top_10_universities = df_2021_sorted.head(10)['university'].tolist()
    return top_10_universities

"""
21. Exploring data manipulation and falsification, continued

This time, let's manipulate the data by changing the source files
instead.
Create a copy of data/2021.csv and name it
data/2021_falsified.csv.
Modify the data in such a way that UC Berkeley comes out on top.

For this part, you will also need to load in the new data
as part of the function.
The function does not take an input; you should get it from the file.

Return the top 10 universities from the falsified data.
"""
import shutil
def q21():
    # TODO
    #make a copy of the 2021 df
    shutil.copy('data/2021.csv', 'data/2021_falsified.csv')
    df = pd.read_csv('data/2021_falsified.csv', encoding='latin-1')

    #manually changing overall score values
    df.loc[df['University'] == 'University of California, Berkeley (UCB)', 'Overall Score'] = 100.0
    print(df[df['University'] == 'University of California, Berkeley (UCB)'])
    df.loc[df['University'] != 'University of California, Berkeley (UCB)', 'Overall Score'] -= 0.5

    #sorting by overall score values
    df_sorted = df.sort_values(by='Overall Score', ascending=False)
    df_sorted.to_csv('data/2021_falsified.csv', index=False, encoding='latin-1')
    top_10_universities = df_sorted.head(10)['University'].tolist()

    return top_10_universities
    # raise NotImplementedError

"""
22. Exploring data manipulation and falsification, continued

Which of the methods above do you think would be the most effective
if you were a "bad actor" trying to manipulate the rankings?

Which do you think would be the most difficult to detect?

=== ANSWER Q22 BELOW ===
I think manually changing the values would be the most difficult to detect because
anyone can go in and see what the formula is for the new_score column, but no one
will have the patience or time to go through each row and make sure the data is accurate.
=== END OF Q22 ANSWER ===
"""

"""
===== Wrapping things up =====

To wrap things up, we have collected
everything together in a pipeline for you
below.

**Don't modify this part.**
It will put everything together,
run your pipeline and save all of your answers.

This is run in the main function
and will be used in the first part of Part 2.
"""

UNFINISHED = 0

def log_answer(name, func, *args):
    try:
        answer = func(*args)
        print(f"{name} answer: {answer}")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},{answer}\n')
            print(f"Answer saved to {ANSWER_FILE}")
    except NotImplementedError:
        print(f"Warning: {name} not implemented.")
        with open(ANSWER_FILE, 'a') as f:
            f.write(f'{name},Not Implemented\n')
        global UNFINISHED
        UNFINISHED += 1

def PART_1_PIPELINE():
    open(ANSWER_FILE, 'w').close()

    try:
        dfs = load_input()
    except NotImplementedError:
        print("Welcome to Part 1! Implement load_input() to get started.")
        dfs = []

    # Questions 1-6
    log_answer("q1", q1, dfs)
    log_answer("q2", q2, dfs)
    log_answer("q3a", q3, dfs)
    # 3b: commentary
    log_answer("q4", q4, dfs)
    # 4b: commentary
    log_answer("q5a", q5a, dfs)
    log_answer("q5b", q5b, dfs)
    log_answer("q5c", q5c)
    # 6a: commentary
    # 6b: commentary
    log_answer("q6c", q6c)

    # Questions 7-10
    log_answer("q7", q7, dfs)
    log_answer("q8a", q8a, dfs)
    # 8b: commentary
    log_answer("q9", q9, dfs)
    # 10: avg_2021
    avg_2021 = q10_helper(dfs)
    log_answer("q10", q10, avg_2021)

    # Questions 11-15
    log_answer("q11", q11, avg_2021)
    log_answer("q12", q12a, avg_2021)
    # 12b: commentary
    log_answer("q13", q13a, avg_2021)
    # 13b: commentary
    log_answer("q14a", q14a, avg_2021)
    # 14b: commentary

    # Questions 15-17
    top_10 = q15_helper(dfs)
    log_answer("q15", q15, top_10)
    log_answer("q16", q16, top_10)
    log_answer("q17", q17a, top_10)
    # 17b: commentary

    # Questions 18-20
    log_answer("q18", q18, dfs)
    # 19: commentary

    # Questions 20-22
    log_answer("q20a", q20a, dfs)
    log_answer("q20b", q20b, dfs)
    log_answer("q21", q21)
    # 22: commentary

    # Answer: return the number of questions that are not implemented
    if UNFINISHED > 0:
        print("Warning: there are unfinished questions.")

    return UNFINISHED

"""
That's it for Part 1!

=== END OF PART 1 ===

Main function
"""

if __name__ == '__main__':
    log_answer("PART 1", PART_1_PIPELINE)
