# Multi-Agent System for Flooding Disaster Management

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
4. Download the original dataset from:  https://drive.google.com/drive/folders/1leN9eWVQcvWDVYwNb2GCo5ML_wBEycWD or the modified version from: https://drive.google.com/file/d/1wz_-7DJdcnxspI0Yv-hAwmW0ImT2hYQP/view?usp=share_link

   Put the files into the folder **dataset**.

5. Train the Neural Network model (can be time-consuming):
   ```sh
   cd python
   python experiment.py --mode train
   ```
   You can personalize your experiment by changing the parameter of the NN (see the experiment.py file).

   If you train the model, your model will be saved in:
   ```sh
   python/models/ 
   ```
   Please Note: if you don't want to train the model you can directlypass to the **next point** and refer to the model called **experiment.pt**.

6. Test your model with:
   ```sh
   cd python
   python experiment.py --mode test
   ```
   Your predictions will be saved in:
   ```sh
   python/predictions/ 
   ```

7. Start the MAS:

   ```sh
   cd MAS
   bash startmas.sh 
   ```
  
8. Use Redis to create a connection between the NN and the MAS (for more information check https://redis.io/).

   Turn on Redis:
   ```sh
   redis-cli
   ```
9. Start the interaction between the MAS and the Neural Network: 

   **Send segmented mask to the MAS**:

    ```sh
    cd src
    python detection.py 
    ```
    In another terminal:

     ```sh
     cd MAS/DALI/mas/py
     python redis2MAS.py 
     ```

   **Simulate weather station**: 

     ```sh
     cd src
     python weather_simulator.py 
     ```

     In another terminal:

     ```sh
     cd MAS/DALI/mas/py
     python redis2MAS.py 
     ```

 
