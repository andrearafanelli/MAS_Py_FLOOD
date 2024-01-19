<h1 align="center"> Neural-logic Multi-agent System for flood event detection </h1>
<div>
<img src=https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white>
<img src=https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=for-the-badge&logo=PyTorch>
<img src=https://img.shields.io/badge/redis-%23DD0031.svg?style=for-the-badge&logo=redis&logoColor=white>
</div>



<h2 align="center">Multi-Agent System for Flooding Disaster Management</h2>

<img src= ./ASSETS/image-mas.png class="center" height=500px width=500 px>.

### Reference paper

> Andrea Rafanelli, Stefania Costantini, Giovanni De Gasperis. "[Neural-logic Multi-agent System for Flood Event Detection]", in: Intelligenza Artificiale, 2023.

Bibtex: 
```bibtex
@article{10.3233/IA-230004,
	    author = {Rafanelli, Andrea and Costantini, Stefania and De Gasperis, Giovanni},
	    title = "{Neural-logic Multi-agent System for Flood Event Detection}",
	    booktitle = {Intelligenza Artificiale},
	    publisher = {IOS Press},
	    year = {2023},
	    vol = {17},
	    pages = {19-35}
	    abstract ={This paper shows the capabilities offered by an integrated neural-logic multi-agent system (MAS). Our case study encompasses logical agents and a deep learning (DL) component, to devise a system specialised in monitoring flood events for civil protection purposes. More precisely, we describe a prototypical framework consisting of a set of intelligent agents, which perform various tasks and communicate with each other to efficiently generate alerts during flood crisis events. Alerts are only delivered when at least two separates sources agree on an event on the same zone, i.e. aerial images and severe weather reports. Images are segmented by a neural network trained over eight classes of topographical entities. The resulting mask is analysed by a Logic Image Descriptor (LID) which then submit the perception to a logical agent.,}
	    issn = {2211-0097},
	    doi = {10.3233/IA-230004},
	    url = {https://doi.org/10.3233/IA-230004},
}	
```

-------------------------------------------------------------------------------------

## Info

The Multi-Agent System (MAS) has been developed using DALI, running on top of SICStus Prolog. 

You can download DALI and test it by running an example DALI MAS:
```sh
git clone https://github.com/AAAI-DISIM-UnivAQ/DALI.git
cd DALI/Examples/advanced
bash startmas.sh
```

## Instructions

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

 
