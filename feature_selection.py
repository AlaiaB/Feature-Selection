import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, f_classif, f_regression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
import numpy as np
import os

class FeatureSelection:
    """
    A class used to perform feature selection on a dataset.

    ...

    Attributes
    ----------
    data : DataFrame
        The input data for feature selection.
    target : str
        The target variable in the data.

    Methods
    -------
    calculate_importance(method='rf'):
        Calculates the importance of features using a specified method.
    select_features(importance, thresholds):
        Selects features based on their importance and a list of thresholds.
    rfe(model, cv=10, n_repeats=3, random_state=1, thresholds=np.linspace(0.01, 0.1, 10)):
        Performs Recursive Feature Elimination (RFE) to select features.
    averaged_importance(method='rf', cv=10, n_repeats=3, random_state=1):
        Calculates the averaged importance of features using a specified method.
    filter_var_imp(k='all', problem_type='classification'):
        Filters variable importance based on a specified problem type.
    """

    def __init__(self, data, target):
        """
        Constructs all the necessary attributes for the FeatureSelection object.

        Parameters
        ----------
        data : DataFrame
            The input data for feature selection.
        target : str
            The target variable in the data.
        """

        self.data = data
        self.target = target

    def calculate_importance(self, method='rf'):
        """
        Calculates the importance of features using a specified method.

        Parameters
        ----------
        method : str, optional
            The method to use for calculating feature importance (default is 'rf').

        Returns
        -------
        array
            The feature importances.
        """

        if method == 'rf':
            model = RandomForestClassifier()
            param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
        elif method == 'gbm':
            model = GradientBoostingClassifier()
            param_grid = {'learning_rate': [0.1, 0.01, 0.001], 'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
        else:
            raise ValueError("Invalid method. Expected 'rf' or 'gbm'.")

        if os.path.exists(f'{method}_best_params.joblib'):
            best_params = joblib.load(f'{method}_best_params.joblib')
            model.set_params(**best_params)
        else:
            grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
            grid_search.fit(self.data.drop(self.target, axis=1), self.data[self.target])
            best_params = grid_search.best_params_
            joblib.dump(best_params, f'{method}_best_params.joblib')
            model.set_params(**best_params)

        model.fit(self.data.drop(self.target, axis=1), self.data[self.target])
        return model.feature_importances_

    def select_features(self, importance, thresholds):
        """
        Selects features based on their importance and a list of thresholds.

        Parameters
        ----------
        importance : array
            The importance of features.
        thresholds : list
            The list of thresholds for feature selection.

        Yields
        -------
        array
            The selected features and their corresponding threshold.
        """

        for threshold in thresholds:
            sfm = SelectFromModel(importance, threshold=threshold)
            yield sfm.fit_transform(self.data), threshold

    def rfe(self, model, cv=10, n_repeats=3, random_state=1, thresholds=np.linspace(0.01, 0.1, 10)):
        """
        Performs Recursive Feature Elimination (RFE) to select features.

        Parameters
        ----------
        model : object
            The machine learning model to use for RFE.
        cv : int, optional
            The number of folds in cross-validation (default is 10).
        n_repeats : int, optional
            The number of times to repeat cross-validation (default is 3).
        random_state : int, optional
            The seed for the random number generator (default is 1).
        thresholds : array, optional
            The thresholds for feature selection (default is np.linspace(0.01, 0.1, 10)).

        Returns
        -------
        float
            The best threshold for feature selection.
        """

        rskf = RepeatedStratifiedKFold(n_splits=cv, n_repeats=n_repeats, random_state=random_state)
        best_score = 0
        best_threshold = None
        for X, threshold in self.select_features(self.calculate_importance(), thresholds):
            scores = cross_val_score(model, X, self.data[self.target], cv=rskf, scoring='roc_auc')
            mean_score = np.mean(scores)
            if mean_score > best_score:
                best_score = mean_score
                best_threshold = threshold
        return best_threshold

    def averaged_importance(self, method='rf', cv=10, n_repeats=3, random_state=1):
        """
        Calculates the averaged importance of features using a specified method.

        Parameters
        ----------
        method : str, optional
            The method to use for calculating feature importance ((default is 'rf').
        cv : int, optional
            The number of folds in cross-validation (default is 10).
        n_repeats : int, optional
            The number of times to repeat cross-validation (default is 3).
        random_state : int, optional
            The seed for the random number generator (default is 1).

        Returns
        -------
        float
            The averaged feature importance.
        """

        if method == 'rf':
            model = RandomForestClassifier()
        elif method == 'gbm':
            model = GradientBoostingClassifier()
        else:
            raise ValueError("Invalid method. Expected 'rf' or 'gbm'.")

        rskf = RepeatedStratifiedKFold(n_splits=cv, n_repeats=n_repeats, random_state=random_state)
        scores = cross_val_score(model, self.data.drop(self.target, axis=1), self.data[self.target], cv=rskf, scoring='roc_auc')
        return np.mean(scores)
    
    def filter_var_imp(self, k='all', problem_type='classification'):
        """
        Filters variable importance based on a specified problem type.

        Parameters
        ----------
        k : int or 'all', optional
            The number of top features to select (default is 'all').
        problem_type : str, optional
            The type of problem, either 'classification' or 'regression' (default is 'classification').

        Returns
        -------
        array
            The scores of the selected features.
        """
        
        if problem_type == 'classification':
            score_func = f_classif
        elif problem_type == 'regression':
            score_func = f_regression
        else:
            raise ValueError("Invalid problem_type. Expected 'classification' or 'regression'.")

        selector = SelectKBest(score_func, k=k)
        selector.fit(self.data.drop(self.target, axis=1), self.data[self.target])
        return selector.scores_
