# CARLA_0.9.6

### Environment Setup
CARLA's experimental setup follows the [DBC](https://github.com/facebookresearch/deep_bisim4control) setup.
Download CARLA_0.9.6 from https://github.com/carla-simulator/carla/releases.

Add to your python path:
```
export PYTHONPATH=$PYTHONPATH:/your/path/CARLA_0.9.6/PythonAPI
export PYTHONPATH=$PYTHONPATH:/your/path/CARLA_0.9.6/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:/your/path/CARLA_0.9.6/PythonAPI/carla/dist/carla-0.9.6-py3.5-linux-x86_64.egg
```
and then copy file:
```
copy ./CARLA_0.9.6/carla_env.py to /CARLA_0.9.6/PythonAPI/carla/agents/navigation
```

### Environment Setup
Run MFSC on CARLA 0.9.6:

**Terminal 1:**
```
$ cd CARLA_0.9.6
$ bash CarlaUE4.sh -fps 20
```

**Terminal 2:**
```
$ bash run_local_carla096.sh
```