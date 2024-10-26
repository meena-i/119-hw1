"""
Part 2: Performance Comparisons

**Released: Wednesday, October 16**

In this part, we will explore comparing the performance
of different pipelines.
First, we will set up some helper classes.
Then we will do a few comparisons
between two or more versions of a pipeline
to report which one is faster.
"""

import part1
import time
import matplotlib.pyplot as plt
"""
=== Questions 1-5: Throughput and Latency Helpers ===

We will design and fill out two helper classes.

The first is a helper class for throughput (Q1).
The class is created by adding a series of pipelines
(via .add_pipeline(name, size, func))
where name is a title describing the pipeline,
size is the number of elements in the input dataset for the pipeline,
and func is a function that can be run on zero arguments
which runs the pipeline (like def f()).

The second is a similar helper class for latency (Q3).

1. Throughput helper class

Fill in the add_pipeline, eval_throughput, and generate_plot functions below.
"""

# Number of times to run each pipeline in the following results.
# You may modify this if any of your tests are running particularly slow
# or fast (though it should be at least 10).
NUM_RUNS = 10
import timeit
class ThroughputHelper:
    def __init__(self):
        # Initialize the object.
        # Pipelines: a list of functions, where each function
        # can be run on no arguments.
        # (like: def f(): ... )
        self.pipelines = []

        # Pipeline names
        # A list of names for each pipeline
        self.names = []

        # Pipeline input sizes
        self.sizes = []

        # Pipeline throughputs
        # This is set to None, but will be set to a list after throughputs
        # are calculated.
        self.throughputs = None

    def add_pipeline(self, name, size, func):
        self.pipelines.append(func)
        self.names.append(name)
        self.sizes.append(size)
        #raise NotImplementedError
    
    def compare_throughput(self):
        # Measure the throughput of all pipelines
        # and store it in a list in self.throughputs.
        # Make sure to use the NUM_RUNS variable.
        # Also, return the resulting list of throughputs,
        # in **number of items per second.**
        self.throughputs = []
        for i in range(len(self.pipelines)):
            pipeline = self.pipelines[i]
            size = self.sizes[i]

            execution_time = timeit.timeit(pipeline, number=NUM_RUNS)
            throughput = (NUM_RUNS * size)/ execution_time
            self.throughputs.append(throughput)
        
        return self.throughputs
        #raise NotImplementedError

    def generate_plot(self, filename):
        # Generate a plot for throughput using matplotlib.
        # You can use any plot you like, but a bar chart probably makes
        # the most sense.
        # Make sure you include a legend.
        # Save the result in the filename provided.
        #plt.figure(figsize=(10, 6))
        plt.bar(self.names, self.throughputs)
        plt.xlabel('Pipelines')
        plt.ylabel('Throughput (items/second)')
        plt.title('Throughput Comparison of Pipelines')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        #return filename
        #raise NotImplementedError

"""
As your answer to this part,
return the name of the method you decided to use in
matplotlib.

(Example: "boxplot" or "scatter")
"""

def q1():
    # Return plot method (as a string) from matplotlib
    return 'bar'
    #raise NotImplementedError

"""
2. A simple test case

To make sure your monitor is working, test it on a very simple
pipeline that adds up the total of all elements in a list.

We will compare three versions of the pipeline depending on the
input size.
"""

LIST_SMALL = [10] * 100
LIST_MEDIUM = [10] * 100_000
LIST_LARGE = [10] * 100_000_000

def add_list(l):
    # TODO
    # Please use a for loop (not a built-in)
    total = 0
    for i in l:
        total += i
    return total
    #raise NotImplementedError

def q2a():
    # Create a ThroughputHelper object
    h = ThroughputHelper()
    # Add the 3 pipelines.
    # (You will need to create a pipeline for each one.)
    # Pipeline names: small, medium, large
    def small_pipeline():
        add_list(LIST_SMALL)
    def medium_pipeline():
        add_list(LIST_MEDIUM)
    def large_pipeline():
        add_list(LIST_LARGE)

    h.add_pipeline('small', len(LIST_SMALL), small_pipeline)
    h.add_pipeline('medium', len(LIST_MEDIUM), medium_pipeline)
    h.add_pipeline('large', len(LIST_LARGE), large_pipeline)

    throughputs = h.compare_throughput()
    #raise NotImplementedError
    # Generate a plot.
    # Save the plot as 'output/q2a.png'.
    # TODO
    h.generate_plot('output/q2a.png')
    # Finally, return the throughputs as a list.
    # TODO
    return throughputs
"""
2b.
Which pipeline has the highest throughput?
Is this what you expected?

=== ANSWER Q2b BELOW ===
The medium pipeline has the highest throughput, which is not what I expected. I expected the
smallest dataset to have the highest throughput since it would have the smallest
processing time.
=== END OF Q2b ANSWER ===
"""

"""
3. Latency helper class.

Now we will create a similar helper class for latency.

The helper should assume a pipeline that only has *one* element
in the input dataset.

It should use the NUM_RUNS variable as with throughput.
"""

class LatencyHelper:
    def __init__(self):
        # Initialize the object.
        # Pipelines: a list of functions, where each function
        # can be run on no arguments.
        # (like: def f(): ... )
        self.pipelines = []

        # Pipeline names
        # A list of names for each pipeline
        self.names = []

        # Pipeline latencies
        # This is set to None, but will be set to a list after latencies
        # are calculated.
        self.latencies = None

    def add_pipeline(self, name, func):
        self.pipelines.append(func)
        self.names.append(name)
        #raise NotImplementedError

    def compare_latency(self):
        # Measure the latency of all pipelines
        # and store it in a list in self.latencies.
        # Also, return the resulting list of latencies,
        # in **milliseconds.**
        self.latencies = []
        for i in range(len(self.pipelines)):
            pipeline = self.pipelines[i]
            
            execution_time = timeit.timeit(pipeline, number=NUM_RUNS)
            latency = execution_time / NUM_RUNS * 1000

            self.latencies.append(latency)
        
        return self.latencies
        #raise NotImplementedError

    def generate_plot(self, filename):
        # Generate a plot for latency using matplotlib.
        # You can use any plot you like, but a bar chart probably makes
        # the most sense.
        # Make sure you include a legend.
        # Save the result in the filename provided.
        plt.bar(self.names, self.latencies)
        plt.xlabel('Pipelines')
        plt.ylabel('Latencies (milliseconds)')
        plt.title('Latencies Comparison of Pipelines')
        #plt.legend()
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()
        #raise NotImplementedError

"""
As your answer to this part,
return the number of input items that each pipeline should
process if the class is used correctly.
"""

def q3():
    # Return the number of input items in each dataset,
    # for the latency helper to run correctly.
    return 1
    #raise NotImplementedError

"""
4. To make sure your monitor is working, test it on
the simple pipeline from Q2.

For latency, all three pipelines would only process
one item. Therefore instead of using
LIST_SMALL, LIST_MEDIUM, and LIST_LARGE,
for this question run the same pipeline three times
on a single list item.
"""

LIST_SINGLE_ITEM = [10] # Note: a list with only 1 item

def q4a():
    # Create a LatencyHelper object
    h = LatencyHelper()

    # Add the same pipeline three times with different names
    h.add_pipeline('single_item_pipeline_1', lambda: sum(LIST_SINGLE_ITEM))
    h.add_pipeline('single_item_pipeline_2', lambda: sum(LIST_SINGLE_ITEM))
    h.add_pipeline('single_item_pipeline_3', lambda: sum(LIST_SINGLE_ITEM))
    
    latencies = h.compare_latency()
    
   
    #raise NotImplementedError
    # Generate a plot.
    # Save the plot as 'output/q4a.png'.
    # TODO
    h.generate_plot('output/q4a.png')
    # Finally, return the latencies as a list.
    # TODO
    return latencies
"""
4b.
How much did the latency vary between the three copies of the pipeline?
Is this more or less than what you expected?

=== ANSWER Q1b BELOW ===
Latency varied by about than a thousandth of a millisecond between the three 
copies of the pipeline. I somewhat expected this because all three pipelines used 
the only one list item as their input, so I didn't think it would take a 
noticeable different amount of time.
=== END OF Q1b ANSWER ===
"""

"""
Now that we have our helpers, let's do a simple comparison.

NOTE: you may add other helper functions that you may find useful
as you go through this file.

5. Comparison on Part 1

Finally, use the helpers above to calculate the throughput and latency
of the pipeline in part 1.
"""

# You will need these:
# part1.load_input
# part1.PART1_PIPELINE

def q5a():
    # Return the throughput of the pipeline in part 1.
    #raise NotImplementedError
    h = ThroughputHelper()
    pt1_data = part1.load_input()
    total_length = 0
    for df in pt1_data:
        total_length += len(df)

    print(total_length)
    h.add_pipeline('part1_pipeline', total_length, lambda: part1.PART_1_PIPELINE())
    
    throughput = h.compare_throughput()
    return throughput

def q5b():
    # Return the latency of the pipeline in part 1.
    #raise NotImplementedError
    h = LatencyHelper()

    h.add_pipeline('part1_pipeline', lambda: part1.PART_1_PIPELINE())
    latency = h.compare_latency()

    return latency

"""
===== Questions 6-10: Performance Comparison 1 =====

For our first performance comparison,
let's look at the cost of getting input from a file, vs. in an existing DataFrame.

6. We will use the same population dataset
that we used in lecture 3.

Load the data using load_input() given the file name.

- Make sure that you clean the data by removing
  continents and world data!
  (World data is listed under OWID_WRL)

Then, set up a simple pipeline that computes summary statistics
for the following:

- *Year over year increase* in population, per country

    (min, median, max, mean, and standard deviation)

How you should compute this:

- For each country, we need the maximum year and the minimum year
in the data. We should divide the population difference
over this time by the length of the time period.

- Make sure you throw out the cases where there is only one year
(if any).

- We should at this point have one data point per country.

- Finally, as your answer, return a list of the:
    min, median, max, mean, and standard deviation
  of the data.

Hints:
You can use the describe() function in Pandas to get these statistics.
You should be able to do something like
df.describe().loc["min"]["colum_name"]

to get a specific value from the describe() function.

You shouldn't use any for loops.
See if you can compute this using Pandas functions only.
"""
import pandas as pd
def load_input(filename):
    # Return a dataframe containing the population data
    # **Clean the data here**
    df = pd.read_csv(filename,  encoding='latin-1')
    
    df = df[~df['Code'].isnull()]
    df = df[~df['Code'].str.contains('OWID_WRL')]
   
    return df
    # raise NotImplementedError

def population_pipeline(df):
    # Input: the dataframe from load_input()
    # Return a list of min, median, max, mean, and standard deviation
    grouped_df = df.groupby('Entity').agg(min_yr = ('Year','min'),
                                   max_yr = ('Year','max'),
                                   min_pop = ("Population (historical)",'first'),
                                   max_pop = ("Population (historical)",'last'))
    
    grouped_df = grouped_df[grouped_df['max_yr'] != grouped_df['min_yr']]
    grouped_df['pop_inc'] = (grouped_df['max_pop'] - grouped_df['min_pop'])/(grouped_df['max_yr'] - grouped_df['min_yr'])

    stats = grouped_df['pop_inc'].describe()[['min', '50%', 'max', 'mean', 'std']].tolist()

    return stats
    
    #raise NotImplementedError
    
def q6():
    # As your answer to this part,
    # call load_input() and then population_pipeline()
    df = load_input('/workspaces/119-hw1/data/population.csv')
    # Return a list of min, median, max, mean, and standard deviation
    return population_pipeline(df)

    #raise NotImplementedError

"""
7. Varying the input size

Next we want to set up three different datasets of different sizes.

Create three new files,
    - data/population-small.csv
      with the first 600 rows
    - data/population-medium.csv
      with the first 6000 rows
    - data/population-single-row.csv
      with only the first row
      (for calculating latency)

You can edit the csv file directly to extract the first rows
(remember to also include the header row)
and save a new file.

Make four versions of load input that load your datasets.
(The _large one should use the full population dataset.)
"""

def load_input_small():
    df_small = load_input('/workspaces/119-hw1/data/population.csv').head(600)
    df_small.to_csv('data/population-small.csv', index=False)
    return pd.read_csv('data/population-small.csv')
    return df_small
    #raise NotImplementedError

def load_input_medium():
    df_medium = load_input('/workspaces/119-hw1/data/population.csv').head(6000)
    df_medium.to_csv('data/population-medium.csv', index=False)
    return df_medium
    #raise NotImplementedError

def load_input_large():
    df_large = load_input('/workspaces/119-hw1/data/population.csv')
    df_large.to_csv('data/population-small.csv', index=False)
    return df_large
    #raise NotImplementedError

def load_input_single_row():
    # This is the pipeline we will use for latency.
    df_single = load_input('/workspaces/119-hw1/data/population.csv').head(1)
    df_single.to_csv('data/population-single-row.csv', index=False)
    return df_single
    #raise NotImplementedError

def q7():
    # Don't modify this part
    s = load_input_small()
    m = load_input_medium()
    l = load_input_large()
    x = load_input_single_row()
    return [len(s), len(m), len(l), len(x)]

"""
8.
Create baseline pipelines

First let's create our baseline pipelines.
Create four pipelines,
    baseline_small
    baseline_medium
    baseline_large
    baseline_latency

based on the three datasets above.
Each should call your population_pipeline from Q7.
"""

def baseline_small():
    s = load_input_small()
    return population_pipeline(s)
    #raise NotImplementedError

def baseline_medium():
    m = load_input_medium()
    return population_pipeline(m)
    #raise NotImplementedError

def baseline_large():
    l = load_input_large()
    return population_pipeline(l)
    #raise NotImplementedError

def baseline_latency():
    x = load_input_single_row()
    return population_pipeline(x)
    #raise NotImplementedError

def q8():
    # Don't modify this part
    _ = baseline_medium()
    return ["baseline_small", "baseline_medium", "baseline_large", "baseline_latency"]

"""
9.
Finally, let's compare whether loading an input from file is faster or slower
than getting it from an existing Pandas dataframe variable.

Create four new dataframes (constant global variables)
directly in the script.
Then use these to write 3 new pipelines:
    fromvar_small
    fromvar_medium
    fromvar_large
    fromvar_latency

As your answer to this part;
a. Generate a plot in output/q9a.png of the throughputs
    Return the list of 6 throughputs in this order:
    baseline_small, baseline_medium, baseline_large, fromvar_small, fromvar_medium, fromvar_large
b. Generate a plot in output/q9b.png of the latencies
    Return the list of 2 latencies in this order:
    baseline_latency, fromvar_latency
"""

# TODO
df = load_input('/workspaces/119-hw1/data/population.csv')
POPULATION_SMALL = df.head(600)
POPULATION_MEDIUM = df.head(6000)
POPULATION_LARGE = df.copy()
POPULATION_SINGLE_ROW = df.head(1)

def fromvar_small():
    return population_pipeline(POPULATION_SMALL)
    #raise NotImplementedError

def fromvar_medium():
    return population_pipeline(POPULATION_MEDIUM)
    #raise NotImplementedError

def fromvar_large():
    return population_pipeline(POPULATION_LARGE)
    #raise NotImplementedError

def fromvar_latency():
    return population_pipeline(POPULATION_SINGLE_ROW)
    #raise NotImplementedError

def q9a():
    # Add all 6 pipelines for a throughput comparison
    # Generate plot in ouptut/q9a.png
    # Return list of 6 throughputs
    h = ThroughputHelper()

    h.add_pipeline('baseline_small', len(POPULATION_SMALL), baseline_small)
    h.add_pipeline('baseline_medium', len(POPULATION_MEDIUM), baseline_medium)
    h.add_pipeline('baseline_large', len(POPULATION_LARGE), baseline_large)

    h.add_pipeline('fromvar_small', len(POPULATION_SMALL), fromvar_small)
    h.add_pipeline('fromvar_medium', len(POPULATION_MEDIUM), fromvar_medium)
    h.add_pipeline('fromvar_large', len(POPULATION_LARGE), fromvar_large)

    throughputs = h.compare_throughput()
    h.generate_plot('output/q9a.png')
    return throughputs
    #raise NotImplementedError

def q9b():
    # Add 2 pipelines for a latency comparison
    # Generate plot in ouptut/q9b.png
    # Return list of 2 latencies
    h = LatencyHelper()
    
    h.add_pipeline('baseline_latency', baseline_latency)
    h.add_pipeline('fromvar_latency', fromvar_latency)
    
    latencies = h.compare_latency()

    h.generate_plot('output/q9b.png')
    return latencies
    
    #raise NotImplementedError

"""
10.
Comment on the plots above!
How dramatic is the difference between the two pipelines?
Which differs more, throughput or latency?
What does this experiment show?

===== ANSWER Q10 BELOW =====
The throughput for fromvar_large is extremely large compared to the rest
of the throughputs. However, baseline latency is much larger than fromvar
latency. Throughput definitely differs more, and a larger throughput for
fromvar shows that it is faster to access data from memory, especially 
larger datasets. Fromvar also has smaller latency which supports this
claim. This experiment shows that it is faster to load data from 
variables rather than loading the input from files directly.
===== END OF Q10 ANSWER =====
"""

"""
===== Questions 11-14: Performance Comparison 2 =====

Our second performance comparison will explore vectorization.

Operations in Pandas use Numpy arrays and vectorization to enable
fast operations.
In particular, they are often much faster than using for loops.

Let's explore whether this is true!

11.
First, we need to set up our pipelines for comparison as before.

We already have the baseline pipelines from Q8,
so let's just set up a comparison pipeline
which uses a for loop to calculate the same statistics.

Create a new pipeline:
- Iterate through the dataframe entries. You can assume they are sorted.
- Manually compute the minimum and maximum year for each country.
- Add all of these to a Python list. Then manually compute the summary
  statistics for the list (min, median, max, mean, and standard deviation).
"""
import numpy as np
def for_loop_pipeline(df):
    # Input: the dataframe from load_input()
    # Return a list of min, median, max, mean, and standard deviation
 
    pop_inc_list = [] 

    for country in df['Entity'].unique():  
        country_data = df[df['Entity'] == country]
        
        if len(country_data) > 1: 
            min_year = country_data['Year'].min()
            max_year = country_data['Year'].max()
            min_pop = country_data[country_data['Year'] == min_year]['Population (historical)'].values[0]
            max_pop = country_data[country_data['Year'] == max_year]['Population (historical)'].values[0]

            pop_diff = max_pop - min_pop
            year_diff = max_year - min_year
            
            if year_diff > 0:
                pop_inc_list.append(pop_diff / year_diff)
        elif len(country_data) == 1:
            return ['null']
    
    
    stats = [float(np.min(pop_inc_list)), 
             float(np.median(pop_inc_list)),
             float(np.max(pop_inc_list)),
             float(np.mean(pop_inc_list)),
             float(np.std(pop_inc_list))]
    
    return stats
    
    #raise NotImplementedError

def q11():
    # As your answer to this part, call load_input() and then
    # for_loop_pipeline() to return the 5 numbers.
    # (these should match the numbers you got in Q6.)
    df = load_input('/workspaces/119-hw1/data/population.csv')
    return for_loop_pipeline(df)
    #raise NotImplementedError

"""
12.
Now, let's create our pipelines for comparison.
As before, write 4 pipelines based on the datasets from Q7.
"""

def for_loop_small():
    return for_loop_pipeline(POPULATION_SMALL)
    #raise NotImplementedError

def for_loop_medium():
    return for_loop_pipeline(POPULATION_MEDIUM)
    #raise NotImplementedError

def for_loop_large():
    return for_loop_pipeline(POPULATION_LARGE)
    #raise NotImplementedError

def for_loop_latency():
    return for_loop_pipeline(POPULATION_SINGLE_ROW)
    #raise NotImplementedError

def q12():
    # Don't modify this part
    _ = for_loop_medium()
    return ["for_loop_small", "for_loop_medium", "for_loop_large", "for_loop_latency"]

"""
13.
Finally, let's compare our two pipelines,
as we did in Q9.

a. Generate a plot in output/q13a.png of the throughputs
    Return the list of 6 throughputs in this order:
    baseline_small, baseline_medium, baseline_large, for_loop_small, for_loop_medium, for_loop_large

b. Generate a plot in output/q13b.png of the latencies
    Return the list of 2 latencies in this order:
    baseline_latency, for_loop_latency
"""

def q13a():
    # Add all 6 pipelines for a throughput comparison
    # Generate plot in ouptut/q13a.png
    # Return list of 6 throughputs
    h = ThroughputHelper()

    h.add_pipeline('baseline_small', len(POPULATION_SMALL), baseline_small)
    h.add_pipeline('baseline_medium', len(POPULATION_MEDIUM), baseline_medium)
    h.add_pipeline('baseline_large', len(POPULATION_LARGE), baseline_large)

    h.add_pipeline('for_loop_small', len(POPULATION_SMALL), for_loop_small)
    h.add_pipeline('for_loop_medium', len(POPULATION_MEDIUM), for_loop_medium)
    h.add_pipeline('for_loop_large', len(POPULATION_LARGE), for_loop_large)

    throughputs = h.compare_throughput()
    h.generate_plot('output/q13a.png')
    return throughputs
    #raise NotImplementedError

def q13b():
    # Add 2 pipelines for a latency comparison
    # Generate plot in ouptut/q13b.png
    # Return list of 2 latencies
    h = LatencyHelper()

    h.add_pipeline('baseline_latency', baseline_latency)
    h.add_pipeline('for_loop_latency', for_loop_latency)

    latencies = h.compare_latency()
    h.generate_plot('output/q13b.png')
    return latencies
    #raise NotImplementedError

"""
14.
Comment on the results you got!

14a. Which pipelines is faster in terms of throughput?

===== ANSWER Q14a BELOW =====
For the largest dataset, the baseline pipeline was faster, but for the
small and medium datasets, the for loop pipeline was faster.
===== END OF Q14a ANSWER =====

14b. Which pipeline is faster in terms of latency?

===== ANSWER Q14b BELOW =====
The for loop pipeline is faster in terms of latency because it has a
lower millisecond value.
===== END OF Q14b ANSWER =====

14c. Do you notice any other interesting observations?
What does this experiment show?

===== ANSWER Q14c BELOW =====
This experiment shows that using a for loop is generally more efficient for
smaller datasets than loading the input from file directly, but
using the baseline pipeline was much faster for the largest dataset.
===== END OF Q14c ANSWER =====
"""

"""
===== Questions 15-17: Reflection Questions =====
15.

Take a look at all your pipelines above.
Which factor that we tested (file vs. variable, vectorized vs. for loop)
had the biggest impact on performance?

===== ANSWER Q15 BELOW =====
Using a pandas dataframe variable had the biggest impact on performance in
comparison to loading input from a file because it had consistently higher
throughputs and a lower latency. When comparing the baseline pipeline with
a for loop pipeline, the baseline pipeline was faster for the largest dataset
but not the smaller ones.
===== END OF Q15 ANSWER =====

16.
Based on all of your plots, form a hypothesis as to how throughput
varies with the size of the input dataset.

(Any hypothesis is OK as long as it is supported by your data!
This is an open ended question.)

===== ANSWER Q16 BELOW =====

===== END OF Q16 ANSWER =====

17.
Based on all of your plots, form a hypothesis as to how
throughput is related to latency.

(Any hypothesis is OK as long as it is supported by your data!
This is an open ended question.)

===== ANSWER Q17 BELOW =====

===== END OF Q17 ANSWER =====
"""

"""
===== Extra Credit =====

This part is optional.

Use your pipeline to compare something else!

Here are some ideas for what to try:
- the cost of random sampling vs. the cost of getting rows from the
  DataFrame manually
- the cost of cloning a DataFrame
- the cost of sorting a DataFrame prior to doing a computation
- the cost of using different encodings (like one-hot encoding)
  and encodings for null values
- the cost of querying via Pandas methods vs querying via SQL
  For this part: you would want to use something like
  pandasql that can run SQL queries on Pandas data frames. See:
  https://stackoverflow.com/a/45866311/2038713

As your answer to this part,
as before, return
a. the list of 6 throughputs
and
b. the list of 2 latencies.

and generate plots for each of these in the following files:
    output/extra_credit_a.png
    output/extra_credit_b.png
"""

# Extra credit (optional)

def extra_credit_a():
    raise NotImplementedError

def extra_credit_b():
    raise NotImplementedError

"""
===== Wrapping things up =====

**Don't modify this part.**

To wrap things up, we have collected
your answers and saved them to a file below.
This will be run when you run the code.
"""

ANSWER_FILE = "output/part2-answers.txt"
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

def PART_2_PIPELINE():
    open(ANSWER_FILE, 'w').close()

    # Q1-5
    log_answer("q1", q1)
    log_answer("q2a", q2a)
    # 2b: commentary
    log_answer("q3", q3)
    log_answer("q4a", q4a)
    # 4b: commentary
    log_answer("q5a", q5a)
    log_answer("q5b", q5b)

    # Q6-10
    log_answer("q6", q6)
    log_answer("q7", q7)
    log_answer("q8", q8)
    log_answer("q9a", q9a)
    log_answer("q9b", q9b)
    # 10: commentary

    # Q11-14
    log_answer("q11", q11)
    log_answer("q12", q12)
    log_answer("q13a", q13a)
    log_answer("q13b", q13b)
    # 14: commentary

    # 15-17: reflection
    # 15: commentary
    # 16: commentary
    # 17: commentary

    # Extra credit
    log_answer("extra credit (a)", extra_credit_a)
    log_answer("extra credit (b)", extra_credit_b)

    # Answer: return the number of questions that are not implemented
    if UNFINISHED > 0:
        print("Warning: there are unfinished questions.")

    return UNFINISHED

"""
=== END OF PART 2 ===

Main function
"""

if __name__ == '__main__':
    log_answer("PART 2", PART_2_PIPELINE)
