import os
import json

import numpy as np
from scipy.linalg import subspace_angles
import matplotlib.pyplot as plt

from utils.file_utils import create_directory


class SubspaceEvaluationAnalyzer:
    @staticmethod
    def save_gradient_subspace(subspace, step, module_name, rank, update_proj_gap):
        dir_path = create_directory(f"gradient_subspace/{rank}-{update_proj_gap}")
        with open(f"{dir_path}/{module_name}.jsonl", "a+") as f:
            json.dump({
                'ortho_matrix': subspace.cpu().detach().float().numpy().tolist(),
                'step': step,
            }, f)
            f.write('\n')


    @staticmethod
    def calculate_principal_angles(m1, m2):
        subspace_angles_rad = subspace_angles(m1, m2)
        return np.rad2deg(subspace_angles_rad)


    @staticmethod
    def calculate_grassmann_distance(principal_angles):
        return np.sqrt(np.sum(np.sin(principal_angles)**2))


    @staticmethod
    def calculate_mean_principal_angle(principal_angles):
        return np.mean(principal_angles)


if __name__ == "__main__":
    rank = 128
    update_proj_gap = 100
    dir_path = f"../gradient_subspace/{rank}-{update_proj_gap}"
    for file_name in [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]:
        with open(f'{dir_path}/{file_name}', 'r') as f:
            subspaces = []
            for line in f:
                subspace_data = json.loads(line)
                subspaces.append(subspace_data['ortho_matrix'])

        grassmann_distance = []
        principal_angles_mean = []
        for m1, m2 in zip(subspaces[:-1], subspaces[1:]):
            principal_angles = SubspaceEvaluationAnalyzer.calculate_principal_angles(m1, m2)
            grassmann_distance.append(SubspaceEvaluationAnalyzer.calculate_grassmann_distance(principal_angles))
            principal_angles_mean.append(SubspaceEvaluationAnalyzer.calculate_mean_principal_angle(principal_angles))

        grassmann_distance_image_dir = create_directory(
            f'{dir_path}/images/{rank}-{update_proj_gap}/grassmann-distance')
        mean_principal_angles_image_dir = create_directory(
            f'{dir_path}/images/{rank}-{update_proj_gap}/mean-principal-angles')

        plt.figure(figsize=(15, 10))
        plt.plot(grassmann_distance, 'o-', color='c')
        plt.xlabel("Consequent Subspace Pair")
        plt.ylabel("Grassmann Distance")
        plt.title(file_name)
        plt.savefig(f'{grassmann_distance_image_dir}/{file_name.replace("jsonl", "png")}')
        plt.close()

        plt.figure(figsize=(15, 10))
        plt.plot(principal_angles_mean, 'o-', color='c')
        plt.xlabel("Consequent Subspace Principal Angles Mean")
        plt.ylabel("Principal Angles Mean")
        plt.title(file_name)
        plt.savefig(f'{mean_principal_angles_image_dir}/{file_name.replace("jsonl", "png")}')
        plt.close()
