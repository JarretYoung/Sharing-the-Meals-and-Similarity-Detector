# Sharing-the-Meals
You and your 4 housemates eat breakfast and dinner together every day, and share the load of preparing the meals.
Now it is time for preparing the schedule of who is responsible for preparing each meal in the next n days (numbered 0, 1, . . . , n − 1). Ideally, the five of you would like to divide the task in such a way that each meal is assigned to one person, no person is assigned to prepare both meals of the same day, and each person is assigned to exactly 2n/5 meals (so that the load is perfectly distributed among you).
However, there are some complications:
 - Perhaps, 2n/5 is not an integer number
 - You all have busy schedules and only have time availability to prepare meals on specific times
To solve the problem, we initially collected the data about the time availability of each person. The five persons will be numbered 0, 1, 2, 3, 4. We get as input a list of lists availability. For a person numbered j and day numbered i, availability[i][j] is equal to:
 - 0, if that person has neither time availability to prepare the breakfast nor the dinner during that day
 - 1, if that person only has time availability to prepare the breakfast during that day
 - 2, if that person only has time availability to prepare the dinner during that day
 - 3, if that person has time availability to prepare either the breakfast or the dinner during that day

After some conversations, you agree that a perfect allocation might not be possible, and someone might have to prepare more meals than others. Moreover, you realised that you might have to order some meals from a restaurant. Nevertheless, you want to achieve a fair distribution and not order many meals from restaurants, so you agreed on the following constraints:
 - Every meal is either allocated to exactly one person or ordered from a restaurant.
 - A person is only allocated to a meal for which s/he has time availability to prepare.
 - Every person should be allocated to at least ⌊0.36n⌋ and at most ⌈0.44n⌉ meals.
 - No more than ⌊0.1n⌋ meals should be ordered from restaurants.
 - No person is allocated to both meals of a day. There are no restrictions on ordering both meals of a day if the other constraints are satisfied.
As the computer scientist in the house, your housemates asked you to design a program to do the allocation. And your housemates are fine with any allocation you give to them (even if it favours you) as long as it satisfy the constraints above!
We solved this problem by writing the function allocate(availability) that returns:
- None (i.e., Python NoneType), if an allocation that satisfy all constraints does not exist.
- Otherwise, it returns (breakfast, dinner), where lists breakfast and dinner specify a valid allocation. breakfast[i] = j if person numbered j is allocated to prepare breakfast on day i, otherwise breakfast[i] = 5 to denote that the breakfast will be ordered from a restaurant on that day. Similarly, dinner[i] = j if person numbered j is allocated to prepare dinner on day i, otherwise dinner[i] = 5 to denote that the dinner will be ordered from a restaurant on that day.

Input and Output:
For example:
 - availability = [[2, 0, 2, 1, 2], [3, 3, 1, 0, 0], [0, 1, 0, 3, 0], [0, 0, 2, 0, 3], [1, 0, 0, 2, 1], [0, 0, 3, 0, 2], [0, 2, 0, 1, 0], [1, 3, 3, 2, 0], [0, 0, 1, 2, 1], [2, 0, 0, 3, 0]]
 - allocate(availability)
 - output: ([3, 2, 1, 4, 0, 2, 3, 2, 2, 3], [4, 0, 3, 2, 5, 4, 1, 1, 3, 0])


# Similarity Detector
You are writing a text similarity detector which compares pairs of short text answers submitted by students for some assessment, and detects those submissions which show unusual levels of similarity to other submissions. You can assume that all of the text submissions have been preprocessed to remove any punctuation and non-alphabetic characters other than spaces, and have been converted to all lowercase characters. In other words, the alphabet consists of the 26 lower case characters from a to z, plus the space character.
For this task, I wrote a Python function called **compare_subs(submission1, submission2)** which will use a retrieval data structure to compare two submissions and determine their similarity. 

Input:
Two submissions, **submission1** and **submission2**. Each submission is given in the form of a string containing only characters in the range [a-z] or the space character.
For example:
 - submission1 = ’the quick brown fox jumped over the lazy dog’
 - submission2 = ’my lazy dog has eaten my homework’
 - compare_subs(submission1, submission2)

Output:
The function compare_subs(submission1, submission2) returns a list of findings with three elements:
 - the longest common substring between submission1 and submission2
 - the similarity score for submission1, expressed as the percentage of submission1 that belongs to the longest common substring (rounded to the nearest integer1), and
 - the similarity score for submission2, expressed as the percentage of submission2 that belongs to the longest common substring (rounded to the nearest integer)
For example:
 - submission1 = ’the quick brown fox jumped over the lazy dog’
 - submission2 = ’my lazy dog has eaten my homework’
 - compare_subs(submission1, submission2)
 - output: [’ lazy dog’, 20, 27] 

