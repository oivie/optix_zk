# Simulate a light scattering scene with PlotOptiX
from plotoptix import TkOptiX
from plotoptix.materials import m_clear
from numpy import random

def run_scene():
    rt = TkOptiX()
    rt.set_param(min_accumulation_step=2, max_accumulation_frames=50)
    rt.set_background(0.1, 0.1, 0.1)

    rt.setup_camera("cam1", eye=[0, 0, -10], lookat=[0, 0, 0])
    rt.set_ambient(0.05)

    N = 100
    pos = random.normal(0, 1, (N, 3))
    r = random.uniform(0.1, 0.4, N)

    rt.set_data("particles", pos=pos, r=r, geom="Sphere", mat=m_clear(glass=True))
    rt.start()

if __name__ == "__main__":
    run_scene()
