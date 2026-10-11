#!/usr/bin/env python3
"""
Landing viz (learning): real sim data; controller interfaces present but not coupled.
"""
from state.core import State
from mass_model.mass import MassModel
from simulation.loop import Simulator, SimConfig
from visualization.viz3d_ctrl import CtrlViz3D

def main():
    s=State(pos=[0.0,0.0,80.0]); m=MassModel(dry_mass_kg=300.0,fuel_kg=0.0)
    sim=Simulator(SimConfig(tf=4.0,dt=0.02)); hist=sim.run(s.copy(),m)
    v=CtrlViz3D(hist,target_pos=(2.0,1.0,0.0)); v.show()

if __name__=='__main__': main()
