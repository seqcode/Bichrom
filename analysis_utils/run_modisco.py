"""
Using TF MoDISco from Shrikumar et al. (2019) to interpret motifs at Hox TF binding sites.
    https://arxiv.org/abs/1811.00416
    https://github.com/kundajelab/tfmodisco
Script adapted from here:
https://github.com/kundajelab/tfmodisco/blob/master/examples/simulated_TAL_GATA_deeplearning/TF%20MoDISco%20TAL%20GATA.ipynb
"""

import h5py
import matplotlib
import numpy as np
import sys



# Set the backend
matplotlib.use('Agg')

# MoDISco imports
import modisco

reload(modisco)
import modisco.backend

reload(modisco.backend.tensorflow_backend)
reload(modisco.backend)
import modisco.nearest_neighbors

reload(modisco.nearest_neighbors)
import modisco.affinitymat

reload(modisco.affinitymat.core)
reload(modisco.affinitymat.transformers)
import modisco.tfmodisco_workflow.seqlets_to_patterns

reload(modisco.tfmodisco_workflow.seqlets_to_patterns)
import modisco.tfmodisco_workflow.workflow

reload(modisco.tfmodisco_workflow.workflow)
import modisco.aggregator

reload(modisco.aggregator)
import modisco.cluster

reload(modisco.cluster.core)
reload(modisco.cluster.phenograph.core)
reload(modisco.cluster.phenograph.cluster)
import modisco.value_provider

reload(modisco.value_provider)
import modisco.core

reload(modisco.core)
import modisco.coordproducers

reload(modisco.coordproducers)
import modisco.metaclusterers

reload(modisco.metaclusterers)
import modisco.util

reload(modisco.util)

from collections import Counter
from modisco.visualization import viz_sequence
reload(viz_sequence)
from matplotlib import pyplot as plt

import modisco.affinitymat.core
reload(modisco.affinitymat.core)
import modisco.cluster.phenograph.core
reload(modisco.cluster.phenograph.core)
import modisco.cluster.phenograph.cluster
reload(modisco.cluster.phenograph.cluster)
import modisco.cluster.core
reload(modisco.cluster.core)
import modisco.aggregator
reload(modisco.aggregator)


def run_methods(hyp_grads_array, onehot_data, task_list, results_file):
    # Define the task dictionaries
    task_to_hyp_scores = {}
    task_to_scores = {}

    print task_list

    task_to_hyp_scores[task_list[0]] = hyp_grads_array  # There is ONLY ONE TASK here.
    grad_star_input = hyp_grads_array * onehot_data
    task_to_scores[task_list[0]] = grad_star_input

    # Values used here are the specified defaults except for a final_min_cluster_size = 60
    # The final_min_cluster_size quantifies the min. amount of supporting seq-lets required, default=30
    tfmodisco_results = modisco.tfmodisco_workflow.workflow.TfModiscoWorkflow(
        # Default parameters
        sliding_window_size=21,
        flank_size=10,
        target_seqlet_fdr=0.15,
        seqlets_to_patterns_factory=
        modisco.tfmodisco_workflow.seqlets_to_patterns.TfModiscoSeqletsToPatternsFactory(
            trim_to_window_size=30,
            initial_flank_to_add=10,
            kmer_len=7, num_gaps=3,
            num_mismatches=2,
            final_min_cluster_size=60)
    )(
        task_names=task_list,
        contrib_scores=task_to_scores,
        hypothetical_contribs=task_to_hyp_scores,
        one_hot=onehot_data)
    # Saving the results
    grp = h5py.File(results_file)
    tfmodisco_results.save_hdf5(grp)


if __name__ == '__main__':
    # grads = np.load(sys.argv[1] + 'gradients.npy')
    # grads_star_inp = np.load(sys.argv[1] + 'gradients_star_inp.npy')

    # boundX = np.load(sys.argv[2])

    # print grads.shape
    # print grads_star_inp.shape

    # run_methods(grads, boundX, ['Ascl1'], sys.argv[1] + 'modisco.out')
    hdf5_results = h5py.File(sys.argv[1] + 'modisco.out', "r")

    print("Metaclusters heatmap")
    import seaborn as sns

    activity_patterns = np.array(hdf5_results['metaclustering_results']['attribute_vectors'])[
        np.array(
            [x[0] for x in sorted(
                enumerate(hdf5_results['metaclustering_results']['metacluster_indices']),
                key=lambda x: x[1])])]
    sns.heatmap(activity_patterns, center=0)
    plt.show()

    metacluster_names = [
        x.decode("utf-8") for x in
        list(hdf5_results["metaclustering_results"]
             ["all_metacluster_names"][:])]

    all_patterns = []

    for metacluster_name in metacluster_names:
        print(metacluster_name)
        metacluster_grp = (hdf5_results["metacluster_idx_to_submetacluster_results"]
        [metacluster_name])
        print("activity pattern:", metacluster_grp["activity_pattern"][:])
        all_pattern_names = [x.decode("utf-8") for x in
                             list(metacluster_grp["seqlets_to_patterns_result"]
                                  ["patterns"]["all_pattern_names"][:])]
        if (len(all_pattern_names) == 0):
            print("No motifs found for this activity pattern")
        for pattern_name in all_pattern_names:
            print(metacluster_name, pattern_name)
            all_patterns.append((metacluster_name, pattern_name))
            pattern = metacluster_grp["seqlets_to_patterns_result"]["patterns"][pattern_name]
            print("total seqlets:", len(pattern["seqlets_and_alnmts"]["seqlets"]))
            background = np.array([0.27, 0.23, 0.23, 0.27])
            print("Task 0 hypothetical scores:")
            viz_sequence.plot_weights(pattern["Ascl1_hypothetical_contribs"]["fwd"])
            print("Task 0 actual importance scores:")
            viz_sequence.plot_weights(pattern["Ascl1_contrib_scores"]["fwd"])
            print("Task 1 hypothetical scores:")
            viz_sequence.plot_weights(pattern["task1_hypothetical_contribs"]["fwd"])
            print("Task 1 actual importance scores:")
            viz_sequence.plot_weights(pattern["task1_contrib_scores"]["fwd"])
            print("Task 2 hypothetical scores:")
            viz_sequence.plot_weights(pattern["task2_hypothetical_contribs"]["fwd"])
            print("Task 2 actual importance scores:")
            viz_sequence.plot_weights(pattern["task2_contrib_scores"]["fwd"])
            print("onehot, fwd and rev:")
            viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["fwd"]),
                                                            background=background))
            viz_sequence.plot_weights(viz_sequence.ic_scale(np.array(pattern["sequence"]["rev"]),
                                                            background=background))

    hdf5_results.close()