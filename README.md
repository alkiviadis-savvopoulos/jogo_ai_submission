# jogo_ai_submission
My dashboard submission for the Jogo AI Data scientist position. This repository will be deleted after evaluation.

# Jogo AI
## _Alkiviadis Savvopoulos - Submission_

My dashboard submission for the position of Data Scientist in Jogo AI. 

Deployed online at : https://jogo-ai-submission-savvopoulos.herokuapp.com/

Alternatively, follow the instructions below :
## How to Run : 
- Install Python ( https://www.python.org/downloads/ )
- If pip is not installed :
    - Run the following to acquire get-pip.py file: 
         ```bash
            curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
        ```
    - Run the following to install pip :
        ```bash
        python get-pip.py
        ```
- Navigate in the submission folder where the _requirements.txt_ file is, and install all dependencies with the following command : 
    ```bash
    pip install -r requirements.txt
    ```
- In the same folder, run the following command to deploy the dashboard locally :
    ```bash
    streamlit run app.py
    ```

## Features

The data available consisted of pose coordinates of various exercises, as they were captured with computer vision algorithms. In the application of the submission, we will explore various key performance indexes regarding the Squatting exercise, the athlete's performance regarding the success of their endeavors as well as the health of the procedure.

The layout of the application consists of the following :

1. A drop-down menu where the end user can select and analyze one of the available squatting sessions 
    - The menu is fed with a number of available _.csv_ files, one for each squatting session available. By selecting a different session, the rest of the dashboard changes.
2. A mini sub-dashboard with key performance indexes 
    - Attempted Repetitions : With the help of a function for locating the local minima of a specific order, we determine the number of repetitions that the athlete attempted to perform.
    - The healthy Range-of-motion is a concept in squatting that is determined from the angle between the hip, the knee, and the heel. The optimal range according to sports science literature is 80° with a variance of 10°. Thus, this indicator shows the number of repetitions which were not subpar ( meaning a small knee angle, thus inefficient training ), while simultaneously not overextending the knee and risking to be injured after performing an unhealthy ROM repeatedly. [^fn1]
    - The final gauge of the mini dashboard shows the average knee degree across all repetitions, and shows in an immersive manner the behaviour of the athlete's squats.
3. An animation of the athlete's best repetition.
    - The best repetition of the session is determined by locating the local minima of all repetitions, and calculating all deviances from the 80° knee angle. After locating the best repetition, an animation is provided with the most crucial squatting coordinates of the body, in order to visualize good behaviour and try to replicate it later on.
4. A time-series of the aforementioned optimal angle, where on the x-axis the whole progression of the session angle is presented. 

## References

[^fn1]: Schoenfeld, Brad J. "Squatting kinematics and kinetics and their application to exercise performance." The Journal of Strength & Conditioning Research 24.12 (2010): 3497-3506.



