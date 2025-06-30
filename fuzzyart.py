"""
    SalientART PY: A Python library of Salient ART. This file is part of Neurol-
    ogy phenotyping.
"""

import os

import numpy as np

class FuzzyART:
    """

    """

    def __init__(self,vigilance,alpha, beta):
        """
        :Param vigilance: vigilance value for training the ART model
        :Param alpha: The parameter for the choice function evaluation
        :Param beta: Learning rate for training the Fuzzy ART model
        """
        self.vigilance = vigilance
        self.alpha = alpha
        self.beta = beta
        self.prototypes = []
        self.labels = []

    def choice(self,input):
        T = []
        for prototype in self.prototypes:
            choice = np.sum(np.minimum(prototype,input))/(self.alpha + np.sum(prototype))
            T.append(choice)
        return T

    def match(self,input):
        M = []
        for prototype in self.prototypes:
            match = np.sum(np.minimum(prototype, input))/np.sum(input)
            M.append(match)
        return M

    def learn(self,input):
        """
        :Param input: the input vector to be fed the ART model
        """
        if len(self.prototypes) == 0:
            # create the first prototype
            self.prototypes.append(input)
            self.labels.append(0)
        else:
            T = self.choice(input)
            M = self.match(input)
            while not all(val==0 for val in T):
                I = T.index(max(T))
                if M[I] >= self.vigilance:
                    self.prototypes[I] =  (1 -self.beta)*self.prototypes[I] + self.beta*np.minimum(input,self.prototypes[I])
                    self.labels.append(I)
                    break
                else:
                    T[I] = -1.0

            if all(val<0 for val in T):
                self.prototypes.append(input)
                self.labels.append(len(self.prototypes)+1)
