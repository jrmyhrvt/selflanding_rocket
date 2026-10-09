# Conventions

## Units
All quantities in SI base-derived units: m, kg, s, rad, N, Pa, K, m/s^2 etc.

## Coordinate System
Local tangent frame (ENU/vertical): 
- X: East
- Y: North  
- Z: Up (away from Earth center locally; for small heights flat-Earth approximation acceptable initially)

## Attitude
Quaternions q = [w, x, y, z] with scalar-first convention (common in many libraries). Euler angles (roll, pitch, yaw) for visualization only. Rotation matrix R(q) maps body -> inertial.

## State Separation
- true_state: ground truth from physics
- sensor_readings: noisy/delayed/corrupted measurements
- estimate: output of estimator with covariance/uncertainty

## Time
Discrete fixed timestep dt by default; events (failures) logged with simulation time t.

## Integration
- Translation: Euler-Cromer
- Time: fixed dt, events logged by simulation time

## Imports
- Internal modules use absolute imports from package root when running with PYTHONPATH=.
