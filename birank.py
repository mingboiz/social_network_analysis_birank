import pandas as pd
import numpy as np
import scipy
import scipy.sparse as spa
from scipy.sparse.linalg import inv, eigs


class Bipartite:
    """
    Class for handling bipartite networks using scipy's sparse matrix
    """

    def __init__(self):
        pass

    def set_edgelist(self, df, parties_col, claims_col, weight_col=None):
        """
        Method to set the edgelist.

        Input:
            df::pandas.DataFrame: the edgelist with at least two columns
            parties_col::string: column of the edgelist dataframe for Involved Parties nodes
            claims_col::string: column of the edgelist dataframe for Claims nodes
            weight_col::string: column of the edgelist dataframe for edge weights

        The edgelist should be represented by a dataframe.
        The dataframe needs at least two columns for the Involed Parties nodes and
        Claims nodes. An optional column can carry the edge weight.
        You need to specify the columns in the method parameters.
        """
        self.df = df.copy()
        self.parties_col = parties_col
        self.claims_col = claims_col
        self.weight_col = weight_col

        self._index_nodes()
        self._generate_adj()

    def generate_degree(self):
        """
        This method returns the degree of nodes in the bipartite network
        """
        parties_df = self.df.groupby(self.parties_col)[self.claims_col].nunique()
        parties_df = parties_df.to_frame(name="degree").reset_index()
        claims_df = self.df.groupby(self.claims_col)[self.parties_col].nunique()
        claims_df = claims_df.to_frame(name="degree").reset_index()
        return parties_df, claims_df

    def _index_nodes(self):
        """
        Representing the network with adjacency matrix requires indexing the parties
        and claims nodes
        """

        self.parties_ids = pd.DataFrame(
            self.df[self.parties_col].unique(), columns=[self.parties_col]
        ).reset_index()
        self.parties_ids = self.parties_ids.rename(columns={"index": "parties_index"})

        self.claims_ids = pd.DataFrame(
            self.df[self.claims_col].unique(), columns=[self.claims_col]
        ).reset_index()
        self.claims_ids = self.claims_ids.rename(columns={"index": "claims_index"})

        self.df = self.df.merge(self.parties_ids, on=self.parties_col)
        self.df = self.df.merge(self.claims_ids, on=self.claims_col)

    def _generate_adj(self):
        """
        Generating the adjacency matrix for the birparite network.
        The matrix has dimension: D * P where D is the number of top nodes
        and P is the number of bottom nodes
        """
        if self.weight_col is None:
            # set weight to 1 if weight column is not present
            weight = np.ones(len(self.df))
        else:
            weight = self.df[self.weight_col]
        self.W = spa.coo_matrix(
            (weight, (self.df["parties_index"].values, self.df["claims_index"].values))
        )

    def generate_prior(self, prior=[]):
        """
        This method returns the prior_vector of the adjacency after it's set correctly

        Input:
            W::scipy's sparse matrix: Adjacency matrix of the bipartite network P*C
            prior::list: List of integers of known fradulent claims
                         e.g. Claim1, Claim3 Fraudulent -> ["Claim 1", "Claim 3"]
        """
        claims_col = self.claims_col
        claims_id = self.claims_ids.copy()
        claims_id["prior_from_prior"] = np.where(
            claims_id[claims_col].isin(prior), 1, 0
        )

        mask = claims_id[claims_col].isin(prior)
        prior_df_temp = claims_id[mask]

        prior_vector = np.zeros(len(claims_id))
        indices_to_put_prior = prior_df_temp["claims_index"].values
        np.put(prior_vector, indices_to_put_prior, np.ones(len(indices_to_put_prior)))
        print(
            "No. of known prior fraudulent flags:",
            sum(prior_vector),
            "Length of prior_vector:",
            len(prior_vector),
        )
        return prior_vector

    def generate_birank(
        self,
        normalizer="BiRank",
        alpha=0.85,
        beta=1,
        max_iter=500,
        tol=1.0e-5,
        prior=[],
        verbose=False,
    ):
        """
        Calculate the Fraud Score of bipartite networks directly.
        See paper https://arxiv.org/pdf/2009.08313.pdf
        for details.
        HITS, CoHITS, BGRM, BiRank Normalizer's implemented
        See paper https://ieeexplore.ieee.org/abstract/document/7572089/
        for details.

        Input:
            W::scipy's sparse matrix:Adjacency matrix of the bipartite network P*C
            normalizer::string:Choose which normalizer to use, see the paper for details
            alpha, beta::float:Damping factors for the rows and columns
            max_iter::int:Maximum iteration times. Set max_iter to 0 or 1 to return the prior_vector and prior
            tol::float:Error tolerance to check convergence
            prior::list: List of integers of known fradulent claims
                         e.g. Claim1, Claim3 Fraudulent -> ["Claim 1", "Claim 3"]
            verbose::boolean:If print iteration information

        Output:
             p, c::numpy.ndarray:BiRank scores for Parties and Claims
        """
        W = self.W.copy()
        df = self.df.copy()
        W = W.astype("float", copy=False)
        WT = W.T

        Kp = np.array(W.sum(axis=1)).flatten()
        Kc = np.array(W.sum(axis=0)).flatten()

        # avoid divided by zero issue
        Kp[np.where(Kp == 0)] += 1
        Kc[np.where(Kc == 0)] += 1

        # Normalizing the W weight matrix
        Kp_ = spa.diags(1 / Kp)
        Kc_ = spa.diags(1 / Kc)
        if normalizer == "HITS":
            Sc = WT
            Sp = W
        elif normalizer == "CoHITS":
            Sc = WT.dot(Kp_)
            Sp = W.dot(Kc_)
        elif normalizer == "BGRM":
            Sc = Kc_.dot(WT).dot(Kp_)
            Sp = Sc.T
        elif normalizer == "BiRank":
            Kp_bi = spa.diags(1 / np.lib.scimath.sqrt(Kp))
            Kc_bi = spa.diags(1 / np.lib.scimath.sqrt(Kc))
            Sc = Kc_bi.dot(WT).dot(Kp_bi)
            Sp = Sc.T
        # Sc and Sp are symmetric normalized weight matrix

        # Generate prior vector

        claims_col = self.claims_col
        claims_id = self.claims_ids.copy()
        claims_id["prior_from_prior"] = np.where(
            claims_id[claims_col].isin(prior), 1, 0
        )

        mask = claims_id[claims_col].isin(prior)
        prior_df_temp = claims_id[mask]

        prior_vector = np.zeros(len(claims_id))
        indices_to_put_prior = prior_df_temp["claims_index"].values
        np.put(prior_vector, indices_to_put_prior, np.ones(len(indices_to_put_prior)))
        print(
            "No. of known prior fraudulent flags:",
            sum(prior_vector),
            "Length of prior_vector:",
            len(prior_vector),
        )

        claims_id["prior_from_setting_indices"] = prior_vector

        mask = claims_id["prior_from_setting_indices"] != claims_id["prior_from_prior"]
        if claims_id[mask].shape[0] > 0:
            print("Prior vector not correctly set!")
            print("Returning prior_vector, prior supplied to function")

            return prior_vector, prior

        parties_id = self.parties_ids.copy()
        # p: Parties (0 vector), c: Claims (prior)
        p0 = np.zeros(Kp_.shape[0])
        p_last = p0.copy()
        c0 = prior_vector
        c_last = c0.copy()

        if max_iter == 0 or max_iter == 1:
            print(
                "Earning Stopping Warning: max_iter is {max_iter}".format(
                    max_iter=max_iter
                )
            )
            print("Returning prior_vector")
            return prior_vector

        for i in range(max_iter):
            c = alpha * (Sc.dot(p_last)) + (1 - alpha) * c0
            p = beta * (Sp.dot(c_last)) + (1 - beta) * p0

            if normalizer == "HITS":
                c = c / c.sum()
                p = p / p.sum()

            err_c = np.absolute(c - c_last).sum()
            err_p = np.absolute(p - p_last).sum()
            if verbose:
                print(
                    "Iteration : {}; top error: {}; bottom error: {}".format(
                        i, err_p, err_c
                    )
                )
            if err_c < tol and err_p < tol:
                break
            p_last = p
            c_last = c
        parties_id["birank_score"] = p
        claims_id["birank_score"] = c
        return (
            parties_id[[self.parties_col, "birank_score"]],
            claims_id[[self.claims_col, "birank_score"]],
        )

    def birank_check(self, claims_birank, parties_birank, alpha, prior_vector):
        """
        Returns the correctness of the birank_score using the Theorem 1:
            1. Check eigenvalues are correctly bounded
            2. Check the analytical solution and the code values
        See paper https://ieeexplore.ieee.org/abstract/document/7572089/ for details

        Parameters:
            W::scipy sparse matrix: Adjacency matrix set by the edgelist of the graph
            claims_birank::np.array: Claims Birank Score calculated by the code
            parties_birank::np.array: Parties Birank Score calculated by the Code
            alpha::float: value of alpha used in Birank calculation
            prior_vector::np.array: prior_vector set during the calculation of Birank

        This assumes Beta is set to 1 during calculation
        """

        p = claims_birank  # p vector is the principal eigenvecctor of matrix S^T*S
        u = parties_birank

        W = self.W.copy()
        WT = W.T

        Ku = np.array(W.sum(axis=1)).flatten()
        Kp = np.array(W.sum(axis=0)).flatten()

        # Normalizing the W weight matrix
        Ku_ = spa.diags(1 / Ku)
        Kp_ = spa.diags(1 / Kp)

        # BiRank Normalization
        Ku_bi = spa.diags(1 / np.lib.scimath.sqrt(Ku))
        Kp_bi = spa.diags(1 / np.lib.scimath.sqrt(Kp))
        Sp = Kp_bi.dot(WT).dot(Ku_bi)  # S^T
        Su = Sp.T  # S

        ST_S = Sp.dot(Su)
        S_ST = Su.dot(Sp)

        print("Check 1 - whether eigenvalues are correctly bounded\n")

        eigenvalues_p, eigenvectors_p = eigs(alpha * ST_S, k=ST_S.shape[0] - 2)
        eigenvalues_c, eigenvectors_c = eigs(alpha * S_ST, k=S_ST.shape[0] - 2)

        if -alpha <= np.round(np.min(eigenvalues_p), 4) and alpha >= np.round(
            np.max(eigenvalues_p), 4
        ):
            print("Check 1 - Eigenvalues of Parties Birank are correctly within bounds")
        else:
            print("Parties Birank wrongly calculated")
            print(
                "Min eigenvalues_p:",
                np.round(np.min(eigenvalues_p), 4),
                "Max eigenvalues_p",
                np.round(np.max(eigenvalues_p), 4),
            )

        if -alpha <= np.round(np.min(eigenvalues_c), 4) and alpha >= np.round(
            np.max(eigenvalues_c), 4
        ):
            print("Check 1 - Eigenvalues of Claims Birank are correctly within bounds")
        else:
            print("Claims Birank wrongly calculated")
            print(
                "Min eigenvalues_c:",
                np.round(np.min(eigenvalues_c), 4),
                "Max eigenvalues_c",
                np.round(np.max(eigenvalues_c), 4),
            )

        print(
            "Check 2 - whether analytical birank_score matches birank_score by code\n"
        )

        claims_soln = inv((spa.identity(ST_S.shape[0]) - alpha * ST_S).tocsc()) * (
            (1 - alpha) * prior_vector
        )
        parties_soln = inv((spa.identity(S_ST.shape[0]) - alpha * S_ST).tocsc()) * (
            (1 - alpha) * Su * prior_vector
        )
        if np.array_equal(np.round(parties_soln, 4), np.round(parties_birank, 4)):
            print("Check 2 - Parties Birank are correctly calculated")
        else:
            print(
                "Parties Birank wrongly calculated, returning parties_birank, parties_soln"
            )
            return parties_birank, parties_soln
        if np.array_equal(np.round(claims_soln, 4), np.round(claims_birank, 4)):
            print("Check 2 - Claims Birank are correctly calculated")
        else:
            print(
                "Claims Birank wrongly calculated, returning claims_birank, claims_soln"
            )
            return claims_birank, claims_soln
