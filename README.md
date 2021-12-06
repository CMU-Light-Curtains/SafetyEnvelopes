# Active Safety Envelopes using Light Curtains with Probabilistic Guarantees

*Project Website*: https://siddancha.github.io/projects/active-safety-envelopes-with-guarantees

<p align="left">
      <img src="media/safety_envelope.gif" width="500px">
</p>

This is the official code for our paper:

>Siddharth Ancha, Gaurav Pathak, Srinivasa Narasimhan, and David Held.<br>
*[Active Safety Envelopes using Light Curtains with Probabilistic Guarantees.](http://www.roboticsproceedings.org/rss17/p045.pdf)*<br>
In Proceedings of **Robotics: Science and Systems (RSS)**, July 2021.


## Installation

#### 1. Clone repository
```bash
git clone --recursive git@github.com:CMU-Light-Curtains/SafetyEnvelopes.git
cd SafetyEnvelopes
```
#### 2. Install python package dependencies:
Note: this repository uses Python 3.6+.
- [fire](https://github.com/google/python-fire)
- [easydict](https://pypi.org/project/easydict/)
- [sacred](https://sacred.readthedocs.io/en/stable/)
- [stable-baseline3](https://stable-baselines3.readthedocs.io/en/master/)
- [flask](https://flask.palletsprojects.com/en/master/)
- [ZeroMQ](https://zeromq.org/languages/python/)

```bash
pip install fire easydict sacred stable-baselines3 Flask scikit-image pyzmq
```

#### 3. Set up the light curtain simulator package [`Simulator`](https://github.com/CMU-Light-Curtains/Simulator).
```bash
cd Simulator
mkdir build && cd build
ANACONDA_ENV_PATH=/path/to/anaconda/env
cmake -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE:FILEPATH=$ANACONDA_ENV_PATH/bin/python \
      -DPYTHON_LIBRARY=$ANACONDA_ENV_PATH/lib/libpython3.so \
      -DPYTHON_INCLUDE_DIR=$ANACONDA_ENV_PATH/include/python3.7m ..
make -j
cd ../..
```

#### 4. Set up the light curtain planning and optimization package [`ConstraintGraph`](https://github.com/CMU-Light-Curtains/ConstraintGraph).
```bash
cd ConstraintGraph
mkdir build && cd build
ANACONDA_ENV_PATH=/path/to/anaconda/env
cmake -DCMAKE_BUILD_TYPE=Release \
      -DBIND_PY=ON \
      -DPYTHON_EXECUTABLE:FILEPATH=$ANACONDA_ENV_PATH/bin/python \
      -DPYTHON_LIBRARY=$ANACONDA_ENV_PATH/lib/libpython3.so \
      -DPYTHON_INCLUDE_DIR=$ANACONDA_ENV_PATH/include/python3.7m ..
make -j
cd ../..
```

#### 5. Install cpp utils.
```bash
cd cpp
mkdir build && cd build
ANACONDA_ENV_PATH=/path/to/anaconda/env
cmake -DCMAKE_BUILD_TYPE=Release \
      -DPYTHON_EXECUTABLE:FILEPATH=$ANACONDA_ENV_PATH/bin/python \
      -DPYTHON_LIBRARY=$ANACONDA_ENV_PATH/lib/libpython3.so \
      -DPYTHON_INCLUDE_DIR=$ANACONDA_ENV_PATH/include/python3.7m ..
make -j
cd ../..
```

#### 6. Set up environment variables
Note: `$DATADIR/synthia` should be the location of the [SYNTHIA-AL](http://synthia-dataset.net/downloads/) dataset.
```bash
# Ideally add these to your .bashrc file
export DATADIR=/path/to/data/dir
export SE_DIR=/path/to/SafetyEnvelopes
export PYTHONPATH=$PYTHONPATH:$SE_DIR:$SE_DIR/Simulator/python:$SE_DIR/ConstraintGraph/apis/py:$SE_DIR/cpp/build
```

#### 7. Set up the dataset
```bash
python data/synthia.py create_video_infos_file --split=mini_train  # creates video info file in $DATADIR 
```

## Running the visualizer
In separate windows, run:

#### 1. Backend server
```bash
python ./visualizer/backend/backend.py --actor=ACTOR_NAME --horizon=HORIZON --load_weights_from_run=RUN_ID
```

#### 2. Frontend server
```bash
cd ./visualizer/frontend
python -m http.server
```

#### 3. Open `http://127.0.0.1:8000/` in browser

## Commands to reproduce experiments in the paper

The following set of commands fully reproduces all experiments in our paper.

### Ours

- `python dagger.py with bevnet_all -p -m sacred`
- `python dagger.py with bevnet_all env.use_random_curtain=True -p -m sacred`

### Baselines

#### 1D CNN

- `python dagger.py with cnn1d_all -p -m sacred`
- `python dagger.py with cnn1d_all env.use_random_curtain=True -p -m sacred`

#### 1D GNN

- `python dagger.py with vres_all -p -m sacred`
- `python dagger.py with vres_all env.use_random_curtain=True -p -m sacred`

### Ablations

#### Don't perform forecasting
- Change to `gt_action = prev_frame.annos["se_ranges"].copy()` in line 256 of `envs/env.py`

#### Don't use the base policy as input
- Appropriately modify `policies/actors/nn/networks/bevnet.py`.

#### Reduce the history from 5 to 1

- `python dagger.py with bevnet_all bevnet.history=2 -p -m sacred`
- `python dagger.py with bevnet_all bevnet.history=2 env.use_random_curtain=True -p -m sacred`

### Evaluation

#### Baseline

- `python eval/main.py --actor=BaselineActor --horizon=50 --split=mini_test`
- `python eval/main.py --actor=BaselineActor --horizon=50 --split=mini_test with env.use_random_curtain=True`

#### Ours

- `python eval/main.py --actor=BevNet --horizon=50 --split=mini_test --load_weights_from_run=154`
- `python eval/main.py --actor=BevNet --horizon=50 --split=mini_test --load_weights_from_run=154 with env.use_random_curtain=True`
