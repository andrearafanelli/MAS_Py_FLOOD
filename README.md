# The Multi-Agent System

The Multi-Agent System (MAS) has been developed using DALI, running on top of SICStus Prolog. 

You can download DALI and test it by running an example DALI MAS:
```sh
git clone https://github.com/AAAI-DISIM-UnivAQ/DALI.git
cd DALI/Examples/advanced
bash startmas.sh
```
-------------------------------------------------------------------------------------
# Instructions

1. Install Redis at https://redis.io/
2. Install SISCtus Prolog at https://sicstus.sics.se/
3. Clone this repository: 
```sh
 git clone https://github.com/andrearafanelli/MAS_Py_FLOOD.git
```
4. Train the Neural Network model (it take many time):
  ```sh
   python experiment.py 
   ```
You can directly pass to the next point, or you can personalize your experiment by changing the parameter of the NN (see the run.py file)

4. Start the MAS:

  ```sh
  cd MAS
  bash startmas.sh 
  ```
5. Start the interaction between the MAS and the Neural Network: 

- Send segmented mask to the MAS:

```sh
  cd src
  python detection.py 
  ```
- Simulate weather station: 

```sh
  cd src
  python weather_simulator.py 
  ```
