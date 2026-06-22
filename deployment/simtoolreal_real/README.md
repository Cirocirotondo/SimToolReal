# HOW TO CONTROL THE ROBOT IN THE REAL WORLD

## HAND (Tesollo Delto 5-fingers 20-DoF)

### Switch-on & connect
- First, power the hand using the bench power supply. Set Voltage=24V and max current=10A
- Then, connect it with the Ethernet cable to the PC
- Open Settings > Network > Ethernet (Enxa0cec...) > tesollo (IPv4 Address = 169.254.186.0, IPv6 Address = fe80::dd7a:2975:1a50:d162 and Hardware Address = 0C:37:96:85:9A:30)

Driver:

```bash
cd ~/git/robohand/ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash 

ros2 launch dg5f_driver dg5f_right_driver.launch.py
```

```bash
cd ~/git/robohand/ros_ws
source /opt/ros/humble/setup.bash
source install/setup.bash 

# if you want to do the test
ros2 run dg5f_driver dg5f_right_test.py 
# if you want to ...
ros2 run dg5f_driver dg5f_right_cmd_receiver.py 

```

-----------------------------------------------------------------------------------------------------------------------------------------------------------


## ARM

### Switch-on & connect

- First of all, you have to connect to the robot. To do so, plug in the robot's ethernet cable to your PC. 
- Then, open Settings > Network > Ethernet (Enxa0cec...) > ur5 (select the second one, with IPv4 Address = 192.168.1.10 and IPv6 Address = fe80::c9d:6272:134d:8e7b)
- From the robot tablet controller: select "Remote Control" on the top-right, next to the burger button.

### Low-level controller:
First of all, you have to run the low-level controller that actually moves the robot. Here is the code to do so:
```bash
cd /home/duplo/simone/SimToolReal/deployment/simtoolreal_real
sudo ./impedance_controller pc_ur_new.json
```
This controller, despite the name, is NOT an impedence controller, rather a **cartesian velocity servo with PD behavior**. Basically, it computes the difference between the target position and the current joint position, and it sends a **velocity command** which is proportional to that error. 

It's possible to communicate with this controller **via TCP using the port 5555**. You can send commands using the *sock.send_json()* function, like in this example.
```Cpp
import time
import zmq

ctx = zmq.Context()
sock = ctx.socket(zmq.PUB)
sock.bind("tcp://*:5555")
time.sleep(1.0)

sock.send_json({
    "target_q": [-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
})
```

All the possible commands that you can use are:
```
target_ee_pose: {"target_ee_pose": [x, y, z, qx, qy, qz, qw]} --> controls the EE with cartesian commands
target_q: {"target_q": [q0, q1, q2, q3, q4, q5]} --> controls the joint positions
trajecotry: {
  "time": [0.0, 2.0, 4.0],
  "path": [
    [q0, q1, q2, q3, q4, q5],
    [q0, q1, q2, q3, q4, q5],
    [q0, q1, q2, q3, q4, q5]
  ]
}
reset_force_sensor: {"reset_force_sensor": true}
```

A good example can be found here: [/home/duplo/git/robohand/src/robot_ipc_control/controller/example.py](/home/duplo/git/robohand/src/robot_ipc_control/controller/example.py)




### High-level controller
If you want a super basic controller that just moves the robot between two positions, use this demo file.
```bash
cd /home/duplo/simone/SimToolReal/deployment/simtoolreal_real
uv run ur5_toggle_home_demo.py
```

This second demo file shows how to get the information about the state:
```bash
cd /home/duplo/simone/SimToolReal/deployment/simtoolreal_real
# to just see the state: 
uv run ur5_print_robot_state.py
```


Note: if you want to use the files in the original location:
```bash
cd ~/git/robohand/src/robot_ipc_control/controller/build
sudo ./impedance_controller ../pc_ur_new.json
cd ~/git/robohand
uv run scripts/ur_ik_from_mjx.py
```

To run the policy, you can use this:
```bash
cd /home/duplo/simone/SimToolReal/deployment/simtoolreal_real

# ATTENTION! Before running this command, read the safety procedures here below!
# Dry run: does not send commands to the robot.
uv run python ur5_policy_arm_controller.py

# Step-by-step debug mode.
uv run python ur5_policy_arm_controller.py --debug-step

# Slow, short real-robot run.
uv run python ur5_policy_arm_controller.py --control-hz 1 --max-steps 10 --send-to-robot

# Full real-robot run. ATTENTION! Before running this command, read the safety procedures here below!
uv run python ur5_policy_arm_controller.py --send-to-robot
```

**SAFETY PROCEDURES**
Don't run a policy right away! 
- First, run just a dry run by removing the "--send-to-robot" arg 
- Then, launch it with "--debug-step": this will make the controller proceed one step at a time, while printing some debug information
- Then, run it with decreased frequency by setting "--control-hz 1" (this controls the frequency. Default is 60Hz)
- Also feel free to add the flag "--max-steps 10" (or any other number of steps) to reduce the movement horizon 


If all the previous steps have worked as expected, feel free to proceed.


-----------------------------------------------------------------------------------------------------------------------------------------------------------

## MUJOCO PLAYGROUND
```bash 
cd ~/git/robohand/src/mujoco_playground
uv run learning/run_hw_in_loop.py --env_name TesolloWristCubeReorient --impl=warp --load_checkpoint_path /home/duplo/git/robohand/src/mujoco_playground/logs/TesolloWristCubeReorient-20260312-093547-6cm_cube_less_friction/checkpoints --logging --real_sys

```

-----------------------------------------------------------------------------------------------------------------------------------------------------------

## POSE ESTIMATION
```bash
cd ~/git/robohand/src/tag-pose-estimation
```
