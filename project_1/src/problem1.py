# -------------------------------------------------------------------------
'''
    Problem 1: getting familiar with python and unit tests.
    In this problem, please install python verion 3 and the following package:
        * nose   (for unit tests)

    To install python packages, you can use any python package managment software, such as pip, conda.
    For example, in pip, you could type `pip install nose` in the terminal to install the package.
                    in conda, you could type `conda install nose` in the terminal.

    Then start implementing function swap().
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''


# --------------------------
def Terms_and_Conditions():
    '''
        By submiting this homework or changing this function, you agree with the following terms:
       (1) Not sharing your code/solution with any student before and after the homework due.
            For example,
            sending your code segment to another student, putting your solution online or lending your laptop
            (if your laptop contains your solution or your dropbox automatically sychronize your solution between your
            home computer and your laptop) to another student to work on this homework will violate this term.
       (2) Not using anyone's code in this homework, build your own solution.
            For example,
                using some code segments from another student or online resources due to
                any reason (like too busy recently) will violate this term.
                Changing other people's code as your solution (such as changing the variable names)
                will also violate this term.
       (3) When discussing with any other student about this homework, only discuss high-level ideas or using
            psudo-code. Don't discuss about the solution at the code level.
            For example,
                discussing with another student about the solution of a function (which needs 5 lines of code to solve),
                and then working on the solution "independently", however the code of the two solutions are exactly the
                same, or only with minor differences  (like changing variable names) will violate this term.
      All violations of (1),(2) or (3) will be handled in accordance with the .
            For more details, please visit Provost's Academic Integrity Site: https://www.lehigh.edu/~inprv/faculty/academicintegrity.html
      Historical Data: in one year, we ended up finding 25% of the students in the class violating this term in their
            homework submissions and we handled ALL of these violations according to the Lehigh Academic Honesty Policy.
    '''
    #########################################
    ## CHANGE CODE HERE
    Read_and_Agree = True # if you have read and agree with the term above, change "False" to "True".
    #########################################
    return Read_and_Agree


# --------------------------
def bubblesort(A):
    '''
        Given a disordered list of integers, rearrange the integers in natural order using bubble sort algorithm.
        Input: A:  a list, such as  [2,6,1,4]
        Output: a sorted list
    '''
    for i in range(len(A) - 1):
        for k in range(len(A) - 1, i, -1):
            if (A[k] < A[k - 1]):
                swap(A, k, k - 1)


# --------------------------
def swap(A, i, j):
    '''
        Swap the i-th element and j-th element in list A.
        Inputs:
            A:  a list, such as  [2,6,1,4]
            i:  an index integer for list A, such as  3
            j:  an index integer for list A, such as  0
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    temp = A[i]
    A[i] = A[j]
    A[j] = temp

    #########################################
