import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Any
import copy
import torch


class ClusteredFederatedServer:

    def __init__(self, n_clusters: int = 2):
        # number of clusters to form
        self.n_clusters = n_clusters

        # it stores which agents belong to which cluster
        # looks like:
        # {
        #   0: ["junction_1", "junction_3"],
        #   1: ["junction_2"]
        # }
        self.clusters = {}

        # reverse lookup for fast broadcasting
        # looks like:
        # {
        #   "junction_1": 0,
        #   "junction_2": 1,
        #   "junction_3": 0
        # }
        self.cluster_assignments = {}

    def cluster_agents(self, agent_weights: Dict[str, Dict]):
        """
        Cluster agents based on similarity of their model weights.
        """

        # agent_weights is received from agents
        # looks like:
        # {
        #   "junction_1": {"layer1.weight": W1, "layer1.bias": b1, ...},
        #   "junction_2": {"layer1.weight": W2, "layer1.bias": b2, ...}
        # }

        # handle empty input safely
        if not agent_weights:
            self.clusters = {}
            self.cluster_assignments = {}
            return self.clusters

        # extract agent IDs
        agent_ids = list(agent_weights.keys())
        # ["junction_1", "junction_2", "junction_3"]

        # if agents < clusters, put everyone into one cluster 0
        if len(agent_ids) < self.n_clusters:
            self.clusters = {0: agent_ids}
            self.cluster_assignments = {aid: 0 for aid in agent_ids}
            return self.clusters

        # flatten each agent's model into a 1D vector
        flat_weights = {}
        # will become:
        # {
        #   "junction_1": array([...]),
        #   "junction_2": array([...])
        # }

        for aid in agent_ids:
            weights = agent_weights[aid]

            flat_list = []
            # flat_list holds flattened layers
            # [
            #   layer1.weight.flatten(),
            #   layer1.bias.flatten(),
            #   layer2.weight.flatten()
            #   ...
            # ]

            for k, w in weights.items():
                # ensure tensor/array is valid and non-empty
                if hasattr(w, "flatten") and hasattr(w, "size") and w.size > 0:
                    flat_list.append(np.array(w).flatten())

            # if model had no usable weights, assign dummy vector, prevents KMeans crash
            if not flat_list:
                flat_weights[aid] = np.zeros(1)
            else:
                flat_weights[aid] = np.concatenate(flat_list)

        # make all vectors same length (required by KMeans) by zero padding
        max_len = max(len(v) for v in flat_weights.values())

        X = np.array(
            [
                np.pad(flat_weights[aid], (0, max_len - len(flat_weights[aid])))
                for aid in agent_ids
            ]
        )

        # X looks like:
        # [
        #   [0.12, 0.03, 0.88, 0.00, 0.00], of aid[0] i.e junction_1
        #   [0.10, 0.05, 0.91, 0.77, 0.68], of aid[1] i.e junction_2
        #   [0.11, 0.02, 0.85, 0.00, 0.00], of aid[2] i.e junction_3
        # ]

        # apply KMeans clustering on model vectors
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)

        # labels look like:
        # [0, 1, 0]

        # initialize cluster containers
        self.clusters = {i: [] for i in range(self.n_clusters)}
        self.cluster_assignments = {}

        # assign each agent to its cluster
        for idx, label in enumerate(labels):
            aid = agent_ids[idx]
            self.clusters[label].append(aid)
            self.cluster_assignments[aid] = label

        # final clusters look like:
        # {
        #   0: ["junction_1", "junction_3"],
        #   1: ["junction_2"]
        # }

        return self.clusters

    def aggregate(
        self, agent_weights: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Perform FedAvg inside each cluster and broadcast back.
        """

        cluster_models = {}
        # will store:
        # {
        #   0: aggregated_weights_of_cluster_0,
        #   1: aggregated_weights_of_cluster_1
        # }

        # iterate cluster by cluster
        for cid, agent_ids in self.clusters.items():
            if not agent_ids:
                continue

            # copy structure from one agent
            first_aid = agent_ids[0]
            agg_weights = copy.deepcopy(agent_weights[first_aid])

            # reset weights to zero (FedAvg start)
            for k in agg_weights:
                if isinstance(agg_weights[k], np.ndarray):
                    agg_weights[k] = np.zeros_like(agg_weights[k])
                elif isinstance(agg_weights[k], torch.Tensor):
                    agg_weights[k] = torch.zeros_like(agg_weights[k])

            # sum weights from all agents in cluster
            for aid in agent_ids:
                w = agent_weights[aid]
                for k in w:
                    agg_weights[k] += w[k]

            # average
            N = len(agent_ids)
            for k in agg_weights:
                agg_weights[k] /= N

            # save
            cluster_models[cid] = agg_weights

        # broadcast aggregated model back to each agent
        result = {}
        # looks like:
        # {
        #   "junction_1": cluster_0_weights,
        #   "junction_2": cluster_1_weights,
        #   "junction_3": cluster_0_weights
        # }

        for aid, cid in self.cluster_assignments.items():
            result[aid] = cluster_models[cid]

        return result