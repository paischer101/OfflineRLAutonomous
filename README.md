# Reinforcement Learning for Real-World Autonomous Scenarios

First in order to be able to run our code you must install and activate the provided environment with
    
    conda env create -f envirotnment.yaml
    conda activate rlautonomous
    
**Note** that this environment was created and used in Ubuntu, thus if you use Windows you can create a new environment by

    conda create -n rlautonomous python=3.6
    conda activate rlautonomous
    
and install the required dependencies listed in the *environment.yaml* file.
Further you need to download the environment and the demonstration data of our field study from [this link](https://drive.google.com/drive/folders/1cuqj1ztnfJVjjTxDgWtfVQNT0_r3rqWC?usp=sharing).
There you can also find our Unity3D scenario and configuration files to run our online RL experiments as mentioned in the paper, as well as a binary for UnityHub to install Unity3D on Linux.
For running our online RL variants we kindly refer to the [unity mlagents package](https://github.com/Unity-Technologies/ml-agents). 
**Note** that you will need to change the path to the demonstration data within the configuration files.

After installing and activating the environment you can reproduce our results for Behavioral Cloning by issuing the following command

    python BC.py dqn_cloning --n-steps 15000 --n-runs 5
    python BC.py drqn_cloning --n-steps 15000 --n-runs 5
    
The first command will run behavrioal cloning and the second one its recurrent version.
Results will be written into a newly created *results* folder in the current directory, within which you can find a checkpoint of the model and tensorboard summaries which contain all the metrics.
Further you can look at some visualizations of the model we prepared, including a sample trajectory, learned action distributions of the model for this trajectory, plus a tsne embedding of the representation the model has learned by

    python visualize_model.py <MODELFILE> --checkpoint <PATH TO CHECKPOINT>

For visualizing our behavioral cloning models, simply use *dqn_cloning* or *drqn_cloning* as modelfile and supply the path to the checkpoint.
This will create a new directory *visualizations* within the directory containing the checkpoint. In the *visualizations* folder you can find a tensorboard eventfile in which all visualizations can be observed.
The TSNE projection is stored separately as an image *tsne_proj.png* within the *visualizations* folder.
So far the hyperparameters used for our work are all hardcoded within the code, but can be adjusted manually.

We also included a recently published method for [offline reinforcement learnning](https://arxiv.org/abs/2006.04779) to our framework which can be trained by 

    python CQ_SAC.py dqn --n-steps 15000 --n-runs 5
    
**Note** that this model up to now only works on single transitions and not as a recurrent variant. Also the hyperparameters for this model are not tuned yet, thus it does not yield reasonable results.
However we expect this model to outperform behavioral cloning after hyperparameter tuning, but might also require collection of more human data.  