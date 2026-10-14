#!/usr/bin/env python3
"""
Interactive landing viz (learning simulation, simplified model).

The virtual controller is ACTUALLY coupled to the simulator here: the sim
adds the controller's thrust command to gravity, so the guided motion is
visibly different from free fall. This is NOT a realistic rocket/engine
model and must not be used for real hardware.
"""
import numpy as np
from simulation.scenario import Scenario, run_scenario
from visualization.viz3d_ctrl import CtrlViz3D
from evaluation import landing as landing_eval


H0_M = 25.0
DT_S = 0.05


def simulate(target_xyz):
    """Run the (simplified) coupled simulation for the given target point.

    Reuses simulation.scenario.run_scenario so there is a single code path.
    """
    sc = Scenario(start_height_m=H0_M, dt=DT_S,
                  target_xy=(float(target_xyz[0]), float(target_xyz[1])))
    return run_scenario(sc)


def main():
    target = np.array([3.0, 2.0, 0.0])
    hist = simulate(target)

    res = landing_eval.evaluate_landing(hist, target)
    print(f"guidance coupled: status={res['status']} "
          f"v={res['v_mag']:.3f} m/s dist={res['dist_xy']:.3f} m "
          f"t_end={hist[-1].t:.1f} s")

    v = CtrlViz3D(hist, target_pos=target, guidance_active=True,
                  resimulate=simulate)
    v.show()


if __name__ == '__main__':
    main()
