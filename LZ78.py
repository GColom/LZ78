#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 23:28:33 2020

@author: Giulio Colombini
"""

from numpy import array, ceil, log2, unique, append, int64, random

#Implementation of minimum sample size ValueError
_MINIMUM_SAMPLE_SIZE_ = 5
_INPUT_ERROR_TEMPLATE_ = 'Sample size is below minimum ({:} outcomes). Minimal sample size can be set, at your own risk, via LZ78.set_minimum_size(new_size). Keep in mind that the accuracy of the estimate grows with the sample size.'

#Set the minimum sample size and the associated exception
def set_minimum_size(new_size):
    '''
    Sets the minimum input size, so that an exception is raised 
    if the input is shorter than this limit, if input check is
    not overridden.

    Parameters
    ----------
    new_size : int
        The new minimum size being set.

    Returns
    -------
    None.

    '''
    global _MINIMUM_SAMPLE_SIZE_ 
    _MINIMUM_SAMPLE_SIZE_ = new_size
    global InputError
    InputError = ValueError(_INPUT_ERROR_TEMPLATE_.format(_MINIMUM_SAMPLE_SIZE_))

set_minimum_size(_MINIMUM_SAMPLE_SIZE_)

#Template for a ValueError for not explicitly supported data types
_UNSUPPORTED_KINDS_ = ['f', 'c', 'V', 'O']
_UNSUPPORTED_KIND_TEMPLATE_ = 'The provided data kind ({:}) is not guaranteed to produce sensible output. If you intend to run the algorithm on the data nonetheless, override the input check. For more information on data kind, please refer to the numpy.dtype.kind documentation.'


def erLZ78(sequence, override_input_check = False):
    '''
    Estimate the entropy rate of a stochastic information source from a sample
    of symbols produced by it.

    Parameters
    ----------
    sequence : numpy array or anything castable to a numpy array.
        The sequence of symbols produced by a source whose entropy
        we want to estimate.
    
    override_input_check : bool
        Allows to disable input checking and precautional casting
        to np.array, for quicker computation(?).
        It is set to False by default.

    Returns
    -------
    Source entropy rate estimate.
    '''
    #Input handling
    if not override_input_check:
        sequence = array(sequence)
        if sequence.dtype.kind in _UNSUPPORTED_KINDS_:
            raise ValueError(_UNSUPPORTED_KIND_TEMPLATE_.format(sequence.dtype.kind))
        if sequence.size < _MINIMUM_SAMPLE_SIZE_:
            raise InputError
    
    #Begin algorithm
    #Init lookup dictionary and current sequence under exam.
    if len(sequence) == 1:
        return 0
    table = {array([]).tobytes() : 0}
    current_seq = array([], dtype = int64)
    alpha_size = unique(array(sequence))
    if len(alpha_size) == 1:
        return 0
    adl = ceil(log2(len(alpha_size))) #alphabet description length
    #Loop over the input, add one number at a time, for each iteration one
    #of three things can happen:
    for idx, number in enumerate(sequence):
        current_seq = append(current_seq, number)
        if current_seq.tobytes() in table:
            if idx == len(sequence)-1:
                #1. Known sequence at EOI, we should break.
                table[-1] = -1
                break
            else:
                #2. Known sequence but there are still characters,
                #add another symbol and repeat lookup.
                continue
        else:
            #   3. Unknown sequence, add prefix and last character to output,
            #   then add it as a new prefix.
            table[current_seq.tobytes()] = len(table)
            current_seq = array([])
    return len(table)*(log2(len(table))+adl)/len(sequence)

#Testing functions

#Entropy of a coin toss with probabilities (x, 1-x)
coin_ent = lambda x : - x *log2(x) + (x-1) *log2(1-x) if x and not x == 1 else 0

def test(sample_size = 100000, resolution = 0.1):
    #Evaluate how many points we'll need depending on the desired resolution.
    graining = int(1/resolution)
    #Prepare a series of Bernoulli (i.i.d. (un)fair coin tosses) samples for testing
    battery = [random.uniform(size=sample_size) for i in range(graining+1)]
    battery = [[1 if n < p*resolution else 0 for n in tst] for p, tst in enumerate(battery)]
    #Estimate entropy rates with the Lempel-Ziv Algorithm
    entropy_rates = [erLZ78(tst, override_input_check = True) for tst in battery]
    #Plotting instructions
    plt.grid(True)
    plt.title('Sample size = '+str(sample_size)+' outcomes\n Probability resolution = '+str(resolution), fontsize = 14)
    plt.xlabel('p', fontsize = 14)
    plt.ylabel(r'$H(\chi)$ (bits/outcome)', fontsize = 14)
    plt.plot([resolution * p for p in range(graining+1)], entropy_rates, label = 'LZ 78 estimate')
    plt.plot([resolution * p for p in range(graining+1)], [coin_ent(resolution*p) for p in range(graining+1)], label = 'Computed')
    plt.legend()
    plt.savefig('test.svg')

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    print('TEST ROUTINE')
    test(resolution = 0.1)