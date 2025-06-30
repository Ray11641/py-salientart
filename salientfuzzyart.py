"""
    SalientART PY: A Python library of Salient ART. This file is part of Neurol-
    ogy phenotyping.
"""

import os

import numpy as np


class SalientFuzzyART:
    """
        This class implements the Salient FuzzyART model.
        Reference:
        Meng, Lei, Ah-Hwee Tan, and Chunyan Miao. "Salience-aware adaptive
        resonance theory for large-scale sparse data clustering." Neural
        Networks 120 (2019): 143-157.
    """

    def __init__(self,vigilance, lamb, alpha, delta):
        """
        :Param vigilance: vigilance value for training the ART model
        :Param lambda:
        :Param alpha: The parameter for the choice function evaluation
        :Param beta: Learning rate for training the Fuzzy ART model
        """
        self.vigilance = vigilance
        self.alpha = alpha
        self.lambda = lamb
        self.delta = delta
        self.prototypes = {}
        self.prototypes["Weights"] = []
        self.prototypes["Frequency"] = []
        self.prototypes["Mu"] = []
        self.prototypes["Variance"] = []
        self.prototypes["Vigilance"] = []
        self.prototypes["Counter"] = []

    def choice(self, input):
        nPrototypes = len(self.prototypes["Weights"])
        T = []
        for itr in range(nPrototypes):
            if np.sum(self.prototypes["Weights"][itr]>0)==0:
                activity = 0
            else:
                salience = self.computesalience(itr)
                numerator = np.minimum(self.prototypes["Weights"][itr],input)
                denominator = self.alpha + np.sum(np.minimum(self.prototypes["Weights"][itr],salience))
                activity = np.sum(np.multiply(salience,numerator))/denominator
            T.append(activity)
        return T

    def match(self, input, index):
        """
        :Param input: current input
        :Param index: index of the prototype
        """
        u = np.minimum(input,self.prototypes["Weights"])
        s = self.computesalience(index)
        m = np.sum(np.multiply(u,s))/np.sum(np.multiply(input,s))
        return m

    def computesalience(self, index):
        """
        :Param I: Index for the prototype
        """
        frequency = self.prototypes["Frequency"][index]
        variance = self.prototypes["Variance"][index]
        salience = []
        for ind in range(salience.shape[0]):
            if frequency[ind]==0:
                salience.append(0.)
            else:
                salience.append(self.lambda*frequency[ind] + (1-self.lambda)*np.exp(-np.sqrt(variance[ind])))
        return np.array(salience)

    def computelearningrate(self, input, index):
        """
        :Param input: current input
        :Param index: Index of the winning prototype
        """
        variance = self.prototypes["Variance"][index]
        mu = self.prototypes["Mu"][index]
        theta = []
        for ind in range(variance.shape[0]):
            if variance[ind]==0:
                theta_jm = np.exp(-(9*(input[ind]-mu[ind])**2)/(2*np.minimum(mu[ind] + 0.01, 1-mu[ind])**2 )
            else:
                theta_jm = np.exp(-(input[ind]-mu[ind])**2)/(2*variance[ind]))
            theta.append(theta_jm)
        return np.array(theta)

    def updatefrequency(self, input, index):
        """
        :Param input: Current input to the network
        :Param index: Index of the winning prototype
        """
        nD = self.prototypes["Frequency"][index].shape[0]
        count = self.prototypes["Counter"][index]
        for itr in range(nD):
            if input[itr]==0:
                self.prototypes["Frequency"][index][itr] *= (count-1)/count
            else:
                self.prototypes["Frequency"][index][itr] *= (count-1)/count
                self.prototypes["Frequency"][index][itr] += 1/count

    def updatemeanvariance(self,input,index):
        mu = self.prototypes["Mu"][index]
        count = self.prototypes["Counter"][index]
        if count == 2:
            z = 0
        else:
            z = (count-2)*self.prototypes["Variance"][index]
        self.prototypes["Mu"][index] = mu + (input-mu)/count
        z_hat = z + np.multiply(input - mu, input - self.prototypes["Mu"][index])
        self.prototypes["variance"][index] = (count-1)*z_hat


    def learn(self, input):
        """
        :Param input: the input vector to be fed to the ART model
        """
        if len(self.prototypes["Weights"])==0:
            self.prototypes["weights"].append(input)
            self.prototypes["Frequency"].append(np.ones(input.shape))
            self.prototypes["Mu"].append(input)
            self.prototypes["Variance"].append(np.zeros(input.shape))
            self.prototypes["Vigilance"].append(self.vigilance)
            self.prototypes["Counter"].append(1.0)

        else:
            T = self.choice(input)
            while not all(T>0 for val in T):
                I = T.index(max(T))
                M = self.match(input, I)
                if M < self.prototypes["Vigilance"][I]:
                    T[I] = -1.0
                    self.prototypes["Vigilance"] *= (1-self.delta)
                else:
                    theta = self.computelearningrate(input)
                    v = np.minimum(self.prototypes["Mu"],input)
                    self.prototypes["Weights"][I] = np.multiply(theta,v) + np.multiply(1-theta,self.prototypes["Weights"][I])
                    self.prototypes["Counter"][I] +=1
                    self.updatefrequency(input,I)
                    self.updatemeanvariance(input,I)
                    self.prototypes["Vigilance"][I]*=(1+self.delta)

            if all(val <0 for val in T):
                self.prototypes["Weights"].append(input)
                self.prototypes["Frequency"].append(np.ones(input.shape))
                self.prototypes["Mu"].append(input)
                self.prototypes["Variance"].append(np.zeros(input.shape))
                self.prototypes["Counter"].append(1.0)
                self.prototypes["Viglance"].append(self.vigilance)
