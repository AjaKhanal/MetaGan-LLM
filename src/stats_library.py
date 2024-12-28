import json

import numpy as np
import pandas as pd
from sdmetrics.single_table import CorrelationSimilarity
from sdmetrics.single_table import CategoryCoverage
from sdmetrics.single_table import RangeCoverage
from sdmetrics.single_table import KSComplement
from sdmetrics.single_table import CSTest
from sdmetrics.single_table import ContinuousKLDivergence
from sdmetrics.single_table import DiscreteKLDivergence


class StatsLibrary:
    """
    param : original_data, synthetic_data
    operation:
      1. describe the data
      2. correlation calculation
      3. categorical feature extraction
      4. continuous feature extraction
    """

    def __init__(self, original_data, synthetic_data):

        """
        param : original_data, synthetic_data
        operation : initialize the class
        """

        self.original_data = original_data
        self.synthetic_data = synthetic_data

    def describe(self):

        """
        param : self object
        operation : describe the data
        """
        return f"Original Data\n{self.original_data.describe()}\n\nUndetermined Data\n{self.synthetic_data.describe()}"

    def corr(self):

        """
        param : self object
        operation : correlation calculation
        return : correlation similarity score
        """

        correlation_score = CorrelationSimilarity.compute(
            real_data=self.original_data,
            synthetic_data=self.synthetic_data)

        return correlation_score

    def get_categorical(self):
        """
        param : self object
        operation : categorical feature extraction
        return : categorical_columns
        """

        categorical_columns = []
        for column in self.original_data.columns:
            if self.original_data[column].dtype == 'object':
                categorical_columns.append(column)

        return categorical_columns

    def get_continuous(self):
        """
        param : self object
        operation : continuous feature extraction
        return : continuous_columns
        """
        continuous_columns = []
        for column in self.original_data.columns:
            if self.original_data[column].dtype != 'object':
                continuous_columns.append(column)

        return continuous_columns

    def category_coverage(self):
        """
        param : self object
        operation : category coverage
        return : category coverage score
        """
        categorical_columns = self.get_categorical()

        if len(categorical_columns) == 0:
            return -1

        categorical_data_original = self.original_data[categorical_columns]
        categorical_data_synthetic = self.synthetic_data[categorical_columns]
        category_coverage_score = CategoryCoverage.compute(
            real_data=categorical_data_original,
            synthetic_data=categorical_data_synthetic)

        return category_coverage_score

    def range_coverage(self):
        """
        param : self object
        operation : range coverage
        return : range coverage score
        """
        if len(self.get_continuous()) == 0:
            return -1

        continuous_columns = self.get_continuous()
        continuous_data_original = self.original_data[continuous_columns]
        continuous_data_synthetic = self.synthetic_data[continuous_columns]
        range_coverage_score = RangeCoverage.compute(
            real_data=continuous_data_original,
            synthetic_data=continuous_data_synthetic)

        return range_coverage_score

    def ks_complement(self):
        """
        param : self object
        operation : ks complement
        return : ks complement score
        """
        continuous_columns = self.get_continuous()
        if len(continuous_columns) == 0:
            return -1

        continuous_data_original = self.original_data[continuous_columns]
        continuous_data_synthetic = self.synthetic_data[continuous_columns]

        ks_complement_score = KSComplement.compute(
            real_data=continuous_data_original,
            synthetic_data=continuous_data_synthetic)

        return ks_complement_score

    def cs_test(self):
        """
        param : self object
        operation : cs test
        return : cs test score
        """
        categorical_columns = self.get_categorical()
        if len(categorical_columns) == 0:
            return -1

        categorical_data_original = self.original_data[categorical_columns]
        categorical_data_synthetic = self.synthetic_data[categorical_columns]
        cs_test_score = CSTest.compute(
            real_data=categorical_data_original,
            synthetic_data=categorical_data_synthetic)

        return cs_test_score

    def continuous_kl_divergence(self):
        """
        param : self object
        operation : continuous kl divergence
        return : continuous kl divergence score
        """
        continuous_columns = self.get_continuous()
        if len(continuous_columns) == 0:
            return -1

        continuous_data_original = self.original_data[continuous_columns]
        continuous_data_synthetic = self.synthetic_data[continuous_columns]
        continuous_kl_divergence_score = ContinuousKLDivergence.compute(
            real_data=continuous_data_original,
            synthetic_data=continuous_data_synthetic)

        return continuous_kl_divergence_score

    def discrete_kl_divergence(self):
        """
        param : self object
        operation : discrete kl divergence
        return : discrete kl divergence score
        """
        categorical_columns = self.get_categorical()
        if len(categorical_columns) == 0:
            return -1

        categorical_data_original = self.original_data[categorical_columns]
        categorical_data_synthetic = self.synthetic_data[categorical_columns]
        discrete_kl_divergence_score = DiscreteKLDivergence.compute(
            real_data=categorical_data_original,
            synthetic_data=categorical_data_synthetic)

        return discrete_kl_divergence_score

    def similarity_score(self):
        """
        param : self object
        operation : similarity score
        return : similarity score
        """

        number_of_features = len(self.original_data.columns)
        number_of_categorical_features = len(self.get_categorical())
        number_of_continuous_features = len(self.get_continuous())

        if self.cs_test() == -1 and self.ks_complement() == -1:
            similarity_score = -1
        elif self.cs_test() == -1:
            similarity_score = self.ks_complement()
        elif self.ks_complement() == -1:
            similarity_score = self.cs_test()
        else:
            similarity_score = (number_of_categorical_features / number_of_features) * self.cs_test() + (
                        number_of_continuous_features / number_of_features) * self.ks_complement()

        return similarity_score

    def divergence_score(self):
        """
        param : self object
        operation : divergence score
        return : divergence score
        """
        number_of_features = len(self.original_data.columns)
        number_of_categorical_features = len(self.get_categorical())
        number_of_continuous_features = len(self.get_continuous())
        if self.discrete_kl_divergence() == -1 and self.continuous_kl_divergence() == -1:
            divergence_score = -1
        elif self.discrete_kl_divergence() == -1:
            divergence_score = self.continuous_kl_divergence()
        elif self.continuous_kl_divergence() == -1:
            divergence_score = self.discrete_kl_divergence()
        else:
            divergence_score = (number_of_categorical_features / number_of_features) * self.discrete_kl_divergence() + (
                        number_of_continuous_features / number_of_features) * self.continuous_kl_divergence()

        return divergence_score

    def coverage_score(self):
        """
        param : self object
        operation : coverage score
        return : coverage score
        """
        number_of_features = len(self.original_data.columns)
        number_of_categorical_features = len(self.get_categorical())
        number_of_continuous_features = len(self.get_continuous())

        if self.category_coverage() == -1 and self.range_coverage() == -1:
            coverage_score = -1

        elif self.category_coverage() == -1:
            coverage_score = self.range_coverage()
        elif self.range_coverage() == -1:
            coverage_score = self.category_coverage()
        else:
            coverage_score = (number_of_categorical_features / number_of_features) * self.category_coverage() + (
                        number_of_continuous_features / number_of_features) * self.range_coverage()

        return coverage_score

    def prompt_formation(self):
        categorical_string = ""
        for i in self.get_categorical():
            categorical_string += i + ", "
        categorical_string = categorical_string[:-2]

        continuous_string = ""
        for i in self.get_continuous():
            continuous_string += i + ", "
        continuous_string = continuous_string[:-2]

        results = [self.describe(),
                   "\nKS Test Results for continuous data columns {} is {}:\n".format(continuous_string,
                                                                                      str(
                                                                                          self.ks_complement())),
                   "\nCS Test Results for categorical data columns {} is {}:\n".format(categorical_string,
                                                                                       str(self.cs_test())),
                   "\nJS Divergence for categorical columns {} is {}:\n".format(categorical_string,
                                                                                str(self.discrete_kl_divergence())),
                   "\nJS Divergence for continuous columns {} is {}:\n".format(continuous_string,
                                                                               str(self.continuous_kl_divergence())),
                   "\n Overall Correlation Similarity Score is {}:\n".format(str(self.corr())),
                   "\n Overall Coverage Score is {}:\n".format(str(self.coverage_score())),
                   "\n Overall Similarity Score is {}:\n".format(str(self.similarity_score())),
                   "\n Overall Divergence Score is {}:\n".format(str(self.divergence_score()))]

        return results
