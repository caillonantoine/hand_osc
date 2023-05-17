# Hands to OSC

Super small project to detect hand landmarks using [MediaPipe](https://developers.google.com/mediapipe) before transmission over OSC.

**Warning** raw landmarks are not available, both hands are first re-projected on a position-invariant basis in order to track *only* the relative position of the fingers.


### Usage

```bash
python detect.py
```

Select the camera you want to use using `--device`, and optionally display it using `--show`

```bash
python detect.py --device 2 --show
```

Configure the address and port of the osc target using `--address` and `--port`

```bash
python detect.py --device 2 --address 127.0.0.1 --port 1893
```