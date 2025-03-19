# Spot Tools
A place for spot tools

```bash
python3 -m venv spot_tools_env
pip install -e .
```
To run the linter on everything:
```
pre-commit run --all-files
```


# Examples

You can find an example of the ROS-free spot executor in
`examples/test_spot_executor.py`. You should be able to run this and see
a little plot of the agent moving along a path. Run it with `python -i` so
that the plot stays at the end. If it crashes with an inscrutable error,
you may need to `pip install opencv-python-headless` because of a conflict
between opencv's QT version and matplotlib's QT version.
