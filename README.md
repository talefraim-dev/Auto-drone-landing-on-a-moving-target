Autonomous Drone Landing Project (UE 5.5 + RL)

Project Structure:
1. AirSim_Source: C++ Simulator source code upgraded for UE 5.5 and .NET 8/9.
2. RL_Training: Python training scripts using PPO and Gymnasium.

Setup:
1. AirSim_Source:
Run setup.bat and build.cmd.
Right-click .uproject and select Generate Visual Studio project files.
Build the solution in Visual Studio 2022.

2. RL_Training:
Install dependencies:
pip install gymnasium numpy stable-baselines3 torch cosysairsim

Execution:
1. Start the Unreal Engine project and press Play.
2. Run the training script:
python RL_Training/Run_train.py

The system includes an AUTOSAVE mechanism that automatically loads the latest model from the output folder.