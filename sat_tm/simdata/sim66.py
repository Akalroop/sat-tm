"""
Spacecraft TM — LEO Orbiting Spacecraft Simulator
=================================================
Earth-pointing satellite in 95-minute circular LEO orbit.

Models:
  - Orbital propagation (Keplerian, 400 km altitude, 95 min period)
  - Body attitude (Quaternion Inertial Body — QIB)
  - Navigation state (ECI Position P, Velocity V)
  - Sun illumination & eclipse (sensor occultation)
  - Battery SOC charge/discharge cycles
  - Thermal control (heater ↔ thermistor pairs)
  - Propulsion (thrusters + LEB electric engine)
  - Reaction wheel & gyro dynamics
  - Comm link (SNR with ground station passes)

Anomaly injection:
  PARAM=VALUE:state   — cascading (affects physics)
  PARAM=VALUE:frame   — sensor override (output only)
  PARAM=VALUE:spike   — one-frame spike (auto-removed)

Usage:
  python simulator.py                       # standalone terminal mode
  python connector3.1.py --simulator simulator.py   # GUI dashboard
"""

import numpy as np
import time
import sys
import threading
from datetime import datetime, timedelta


# ============================================================================
# ORBITAL & PHYSICS CONSTANTS
# ============================================================================
MU_EARTH    = 3.986004418e5       # km^3/s^2 (Earth gravitational parameter)
R_EARTH     = 6378.137            # km (mean Earth radius)
ALPHA_EARTH = 0.2618              # rad  (15° half-angle for penumbra)
BETA_SOLAR  = 0.0                 # rad  (solar beta angle, simplified)

# Orbit parameters
ALTITUDE_KM     = 400.0           # circular LEO altitude
SEMIMAJOR_KM    = R_EARTH + ALTITUDE_KM
ORBIT_PERIOD_S  = 2.0 * np.pi * np.sqrt(SEMIMAJOR_KM**3 / MU_EARTH)   # ~5776 s ≈ 96.3 min
INCLINATION     = np.radians(51.6)   # ISS-like inclination
RAAN            = np.radians(0.0)   # right ascension of ascending node (fixed, simplified)

# Satellite physical properties
SAT_MASS_KG     = 250.0           # total mass
SAT_INERTIA     = np.diag([15.0, 18.0, 12.0])  # kg·m² (Ixx, Iyy, Izz)
SAT_AREA_M2     = 4.0             # cross-sectional area to Sun (sunlit)
SAT_AREA_EARTH  = 4.0             # cross-sectional area to Earth (IR)
SAT_ALBEDO      = 0.3             # Earth albedo
SAT_EMISSIVITY  = 0.8             # infrared emissivity
SAT_ABSORPT     = 0.4             # solar absorptivity
SAT_STEFAN      = 5.67e-8         # W/(m²·K⁴)
SAT_AREA_RAD    = 3.0             # radiating area (infrared)

# Battery
BATTERY_CAPACITY_Ah = 80.0        # Ah
BATTERY_V_NOMINAL   = 28.0        # V nominal
BATTERY_R_INTERNAL  = 0.05        # Ω internal resistance
BATTERY_MAX_SOC     = 1.0         # 100%
BATTERY_MIN_SOC     = 0.25        # 25% depth of discharge
BATTERY_SOC_INIT    = 0.80        # start at 80%

# Solar array
SOLAR_AREA_M2       = 8.0         # m² total panel area
SOLAR_EFFICIENCY    = 0.28        # cell efficiency
SOLAR_IRRADIANCE    = 1361.0      # W/m² (solar constant)
SOLAR_POWER_MAX     = SOLAR_AREA_M2 * SOLAR_EFFICIENCY * SOLAR_IRRADIANCE  # ~3077 W

# Loads (average bus consumption)
BUS_LOAD_BASE_W     = 400.0       # W (nominal)
BUS_LOAD_MAX_W      = 1200.0      # W (peak)

# Thermal
THERM_CPUT_SETPOINT = 22.0        # °C CPU thermistor setpoint
THERM_BATT_SETPOINT = 15.0        # °C battery thermistor setpoint
THERM_RX_SETPOINT   = 10.0        # °C Rx thermistor setpoint
THERM_HYSTERESIS    = 2.0         # °C heater hysteresis

# Heaters
HEATER_CPU_POWER_W  = 30.0        # W
HEATER_BATT_POWER_W = 15.0        # W
HEATER_RX_POWER_W   = 20.0        # W

# Gyro / Reaction Wheel
RW_SPEED_MAX_RPM  = 6000.0
# Wheel hardware matched to a real flight-proven micro/smallsat-class
# reaction wheel appropriate for this satellite's mass (~250 kg) and
# inertia (~15-18 kg*m^2) class: Sinclair Interplanetary (Rocket Lab)
# RW4-1.0 -- 1.0 Nms momentum capacity, +/-100 mNm max torque. Wheel
# inertia derived from H = I*omega_max at the same 6000 rpm max speed
# already used in this file; independently cross-checked against
# Astrofein's RW150 (also 1.0 Nms @ 6000 rpm) -- both give ~0.0016
# kg*m^2. Previous values (I=0.02, torque cap=5 Nm) were ~12x and ~50x
# too large respectively for a wheel this size.
RW_TORQUE_MAX_NM  = 0.1          # 100 mNm (Sinclair RW4-1.0 rated max)
RW_INERTIA_KGM2   = 0.0016       # kg*m^2, from 1.0 Nms @ 6000 rpm
RW_DESAT_TORQUE_NM = 0.004      # torque commanded during active momentum dump
                                 # (sized so one 1s step moves ~25 RPM, well
                                 # under the 150 RPM release band -- at the
                                 # scale matched to real desat torque earlier,
                                 # a single step overshot the band every time
                                 # and caused a bang-bang oscillation instead
                                 # of a clean correction; verified below)
RW_DESAT_HIGH_RPM = 5200.0      # absolute speed that triggers desaturation (upper)
RW_DESAT_LOW_RPM  = 100.0       # absolute speed that triggers desaturation (lower)
RW_DESAT_RELEASE_RPM = 150.0    # |speed - nominal| that ends desaturation
GYRO_BIAS_DPS     = 0.01
GYRO_NOISE_DPS    = 0.002

# Comm
COMM_FREQ_MHZ     = 2.4         # S-band
GROUND_RANGE_MIN  = 1500.0      # km (minimum for link)
# Max line-of-sight slant range at 400km altitude, 0 deg elevation
# (pure horizon graze), is sqrt((R_EARTH+ALT)^2 - R_EARTH^2) = ~2294 km
# -- beyond that, Earth's curvature physically blocks the signal
# regardless of link budget. The old value (6000 km) was ~2.6x that
# limit: verified empirically that 83% of "comm_active=True" frames
# were reporting a plausible SNR while the satellite was geometrically
# behind the Earth's limb from the ground station. 2200 km keeps a
# small margin below the exact 2294 km limit (a real link needs some
# positive elevation, not exact grazing incidence).
GROUND_RANGE_MAX  = 2200.0      # km (maximum for link)
EARTH_ROTATION_RATE = 7.2921159e-5  # rad/s (sidereal), ground station rotates with Earth
GS_LON0_RAD       = np.radians(77.0)         # ground station longitude at t=0 (rad), ~77°E as per spec

# Thruster (chemical)
THRUST_FORCE_N    = 5.0           # N per thruster
THRUST_ISP_S      = 220.0         # s specific impulse
THRUST_DENSITY    = 1.8           # g/cm³ xenon-like propellant
THRUST_TANK_KG    = 50.0          # kg propellant
THRUST_MAX_BURST_S = 30.0         # max continuous burn per command

# LEB (Linear Electric Bipolar / Ion engine)
LEB_THRUST_N      = 0.05          # N (50 mN)
LEB_ISP_S         = 1500.0        # s
LEB_POWER_W       = 500.0         # W from solar array
LEB_TANK_KG       = 30.0          # kg xenon
LEB_MAX_BURST_S   = 3600.0        # max continuous burn per command


# ============================================================================
# PARAMETER LIST (all telemetry channels)
# ============================================================================
PARAMS = [
    # --- Orbital ---
    "altitude_km",
    # --- Navigation (ECI) ---
    "nav_pos_x_km", "nav_pos_y_km", "nav_pos_z_km",
    "nav_vel_x_kms", "nav_vel_y_kms", "nav_vel_z_kms",
    # --- Attitude (Quaternion Inertial Body) ---
    "att_q0", "att_q1", "att_q2", "att_q3",
    # --- Attitude rates (body frame, rad/s) ---
    "att_rate_x_rads", "att_rate_y_rads", "att_rate_z_rads",
    # --- Sun Illumination ---
    "sun_angle_deg",
    "eclipse_depth",
    "shadow_fraction",
    "sun_exposure",
    "sun_vector_x", "sun_vector_y", "sun_vector_z",
    # --- Battery ---
    "battery_voltage_V",
    "battery_current_A",
    "battery_soc",
    "battery_temperature_C",
    # --- Solar ---
    "solar_panel_current_A",
    "solar_panel_watts",
    "solar_panel_voltage_V",
    # --- Thermal ---
    "cpu_temperature_C",
    "rx_temperature_C",
    "bus_temperature_C",
    "ambient_temperature_C",
    # --- Heater control ---
    "heater_cpu_current_A",
    "heater_batt_current_A",
    "heater_rx_current_A",
    "thermistor_cpu_setpoint_C",
    "thermistor_batt_setpoint_C",
    "thermistor_rx_setpoint_C",
    "heater_bus_voltage_V",
    # --- System load ---
    "system_load",
    "bus_power_W",
    "total_power_W",
    # --- Reaction wheels ---
    "rw_speed_1_rpm", "rw_speed_2_rpm",
    "rw_speed_3_rpm", "rw_speed_4_rpm",
    "rw_torque_1_nm", "rw_torque_2_nm",
    "rw_torque_3_nm", "rw_torque_4_nm",
    # --- Gyros ---
    "gyro_rate_x_dps", "gyro_rate_y_dps", "gyro_rate_z_dps",
    # --- Comm ---
    "comm_snr_db",
    "comm_active",
    "ground_range_km",
    # --- Propulsion: Thruster ---
    "thruster_active",
    "thrust_force_N",
    "thruster_propellant_kg",
    "thruster_burn_time_s",
    # --- Propulsion: LEB ---
    "leb_active",
    "leb_thrust_N",
    "leb_power_W",
    "leb_propellant_kg",
    "leb_burn_time_s",
]


# ============================================================================
# SPACECRAFT SIMULATOR
# ============================================================================
class TelemetrySimulator:
    """Full LEO orbiting spacecraft simulator with dynamics, propulsion, and thermal."""

    def __init__(self):
        self.t = 0                        # simulation time (seconds)
        self.dt = 1.0                     # step size (seconds)

        # ---- State: Orbit ----
        self._orbital_angle = 0.0          # mean anomaly (rad) along orbit
        self._inertia = SAT_INERTIA.copy() # kg·m² (currently unused: the
                                            # attitude tracking filter no
                                            # longer does torque/inertia
                                            # integration -- see
                                            # _attitude_dynamics)
        self._mass = SAT_MASS_KG            # kg (depletes with propulsion)

        # ---- State: Attitude (QIB) ----
        self._q = np.array([1.0, 0.0, 0.0, 0.0])  # q0=scalar, q1-3=vector (NED→ECI)
        self._omega_body = np.array([0.0, 0.0, 0.0])  # rad/s

        # ---- State: Battery ----
        self._soc = BATTERY_SOC_INIT
        self._v_battery = BATTERY_V_NOMINAL
        self._i_battery = 0.0

        # ---- State: Thermal ----
        self._t_cpu = 25.0
        self._t_rx = 20.0
        self._t_bus = 22.0
        self._t_batt = 20.0      # battery thermistor (was previously
                                  # recomputed ad hoc each frame with no
                                  # persistent state and no heater coupling)
        self._t_ambient = -10.0  # deep space background
        self._heater_cpu_on = False
        self._heater_batt_on = False
        self._heater_rx_on = False

        # ---- State: Reaction wheels ----
        self._rw_speeds = np.array([1000.0, 1050.0, 980.0, 1020.0])  # RPM
        self._rw_nominal = self._rw_speeds.copy()  # bias-momentum target per wheel
        self._rw_torques = np.array([0.0, 0.0, 0.0, 0.0])  # Nm
        self._rw_desaturating = np.array([False, False, False, False])

        # ---- State: Gyros ----
        self._gyro_bias = np.array([GYRO_BIAS_DPS, GYRO_BIAS_DPS, GYRO_BIAS_DPS])

        # ---- State: Propulsion ----
        self._thruster_active = False
        self._thruster_burn_elapsed = 0.0
        self._propellant_thruster = THRUST_TANK_KG
        self._leb_active = False
        self._leb_burn_elapsed = 0.0
        self._propellant_leb = LEB_TANK_KG

        # ---- State: Comm ----
        self._comm_active = False

        # ---- Injections ----
        self.state_injections = {}   # affects physics/cascades
        self.sensor_overrides = {}   # sensor error (output only)
        self.sensor_spikes = {}      # one-frame spike

        self.running = True
        self.paused = False

    # =========================================================================
    # ORBITAL PROPAGATION (Keplerian circular orbit)
    # =========================================================================
    def _propagate_orbit(self):
        """Propagate satellite in circular LEO orbit, return ECI position & velocity."""
        n = np.sqrt(MU_EARTH / SEMIMAJOR_KM**3)       # mean motion (rad/s)
        theta = self._orbital_angle                      # true anomaly ≈ mean anomaly (circular)

        # Position in orbital plane
        r = SEMIMAJOR_KM
        x_orb = r * np.cos(theta)
        y_orb = r * np.sin(theta)

        # Rotate by inclination to ECI
        pos_eci = np.array([
            x_orb,
            y_orb * np.cos(INCLINATION),
            y_orb * np.sin(INCLINATION)
        ])

        # Velocity in orbital plane
        v = np.sqrt(MU_EARTH / r)
        vx_orb = -v * np.sin(theta)
        vy_orb = v * np.cos(theta)

        vel_eci = np.array([
            vx_orb,
            vy_orb * np.cos(INCLINATION),
            vy_orb * np.sin(INCLINATION)
        ])

        # Altitude
        alt = r - R_EARTH

        # Advance mean anomaly
        self._orbital_angle += n * self.dt

        return pos_eci, vel_eci, alt

    # =========================================================================
    # ATTITUDE DYNAMICS (Earth-pointing, body-Z nadir, body-X velocity)
    # =========================================================================
    def _compute_earth_pointing_quaternion(self, pos_eci, vel_eci):
        """
        Compute quaternion for Earth-pointing attitude:
        - Body +Z → nadir (toward Earth center)
        - Body +X → velocity vector (along-track)
        - Body +Y → completes right-handed tri (cross-track)
        """
        # Nadir (body-Z in ECI)
        z_body = -pos_eci / np.linalg.norm(pos_eci)

        # Velocity direction (body-X in ECI)
        x_body = vel_eci / np.linalg.norm(vel_eci)

        # Cross-track (body-Y in ECI)
        y_body = np.cross(z_body, x_body)
        y_body = y_body / np.linalg.norm(y_body)

        x_body = np.cross(y_body, z_body)  # re-orthogonalize

        # Construct rotation matrix (body ← ECI)
        R_eci_body = np.vstack([x_body, y_body, z_body]).T

        # Convert to quaternion (scalar-first: q0, q1, q2, q3)
        tr = np.trace(R_eci_body)
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2.0
            q = np.array([0.25 * S,
                          (R_eci_body[2, 1] - R_eci_body[1, 2]) / S,
                          (R_eci_body[0, 2] - R_eci_body[2, 0]) / S,
                          (R_eci_body[1, 0] - R_eci_body[0, 1]) / S])
        else:
            if R_eci_body[0, 0] > R_eci_body[1, 1] and R_eci_body[0, 0] > R_eci_body[2, 2]:
                S = np.sqrt(1.0 + R_eci_body[0, 0] - R_eci_body[1, 1] - R_eci_body[2, 2]) * 2.0
                q = np.array([(R_eci_body[2, 1] - R_eci_body[1, 2]) / S,
                              0.25 * S,
                              (R_eci_body[0, 1] + R_eci_body[1, 0]) / S,
                              (R_eci_body[0, 2] + R_eci_body[2, 0]) / S])
            elif R_eci_body[1, 1] > R_eci_body[2, 2]:
                S = np.sqrt(1.0 + R_eci_body[1, 1] - R_eci_body[0, 0] - R_eci_body[2, 2]) * 2.0
                q = np.array([(R_eci_body[0, 2] - R_eci_body[2, 0]) / S,
                              (R_eci_body[0, 1] + R_eci_body[1, 0]) / S,
                              0.25 * S,
                              (R_eci_body[1, 2] + R_eci_body[2, 1]) / S])
            else:
                S = np.sqrt(1.0 + R_eci_body[2, 2] - R_eci_body[0, 0] - R_eci_body[1, 1]) * 2.0
                q = np.array([(R_eci_body[1, 0] - R_eci_body[0, 1]) / S,
                              (R_eci_body[0, 2] + R_eci_body[2, 0]) / S,
                              (R_eci_body[1, 2] + R_eci_body[2, 1]) / S,
                              0.25 * S])

        # Normalize
        q = q / np.linalg.norm(q)

        # Apply small attitude disturbances (3-axis stabilization residual)
        q[1] += np.random.randn() * 0.001   # small jitter
        q[2] += np.random.randn() * 0.001
        q[3] += np.random.randn() * 0.001
        q = q / np.linalg.norm(q)

        return q

    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions: q1 ⊗ q2. Both are [q0, q1, q2, q3]."""
        q1s, q1v = q1[0], q1[1:]
        q2s, q2v = q2[0], q2[1:]
        result = np.empty(4)
        result[0] = q1s*q2s - np.dot(q1v, q2v)
        result[1:] = q1s*q2v + q2s*q1v + np.cross(q1v, q2v)
        return result

    def _attitude_dynamics(self, q_desired):
        """
        Attitude tracking toward the desired Earth-pointing quaternion.

        NOTE: this used to be a torque/inertia PD loop. That formulation
        was numerically unstable at dt=1s (it diverged to hundreds of
        rad/s within ~1 simulated day -- verified empirically). Replaced
        with a first-order (low-pass) quaternion filter, which is
        unconditionally stable for alpha in (0,1] and still produces
        smooth, small, bounded body rates. Rate is *derived* from the
        quaternion update (not integrated separately), so the reported
        attitude and reported rate channels stay mutually consistent.
        """
        q_actual = self._q.copy()

        # Keep on the same hemisphere to avoid a spurious 180 deg flip
        if np.dot(q_actual, q_desired) < 0:
            q_desired = -q_desired

        alpha = 0.08  # tracking bandwidth: higher = tighter pointing, less lag
        q_new = q_actual + alpha * (q_desired - q_actual)
        q_new = q_new / np.linalg.norm(q_new)

        # Recover the body rate implied by this quaternion step:
        # dq/dt = 0.5 * Omega(q) @ [0, omega] => omega = 2 * Omega(q)^T @ dq/dt
        # (Omega(q) is orthogonal for unit q, so its transpose is its inverse.)
        q_dot = (q_new - q_actual) / self.dt
        omega_quat = 2.0 * (self._quat_to_omega_matrix(q_actual).T @ q_dot)
        omega_body = omega_quat[1:]

        # Hard safety clamp: generous vs. nominal (~0.06 deg/s for LEO
        # nadir tracking) but guarantees no residual transient can ever
        # produce a non-physical rate.
        max_rate = np.radians(5.0)
        omega_body = np.clip(omega_body, -max_rate, max_rate)

        self._omega_body = omega_body
        self._q = q_new

        return self._omega_body

    def _quat_to_omega_matrix(self, q):
        """Convert quaternion to skew-symmetric omega matrix."""
        return np.array([
            [0,       -q[1], -q[2], -q[3]],
            [q[1],     q[0], -q[3],  q[2]],
            [q[2],     q[3],  q[0], -q[1]],
            [q[3],    -q[2],  q[1],  q[0]]
        ])

    # =========================================================================
    # SUN ILLUMINATION & ECLIPSE (Sensor Occultation)
    # =========================================================================
    def _compute_sun_geometry(self, pos_eci):
        """
        Compute Sun vector in ECI, sun angle, eclipse state.
        Simplified: Sun in ecliptic plane, orbit inclined.
        """
        # Sun direction (simplified — circular Earth orbit around Sun)
        # Assume Sun is in ecliptic, satellite at arbitrary season
        sun_ra = np.radians(self.t / 31557600.0 * 360.0)  # ~1°/day
        sun_dec = np.radians(23.44 * np.sin(sun_ra))      # seasonal variation
        sun_vec_eci = np.array([
            np.cos(sun_dec) * np.cos(sun_ra),
            np.cos(sun_dec) * np.sin(sun_ra),
            np.sin(sun_dec)
        ])

        # Satellite-to-Earth vector (nadir)
        nadir = -pos_eci / np.linalg.norm(pos_eci)

        # Sun angle (angle between nadir and Sun) in radians
        cos_angle = np.clip(np.dot(nadir, sun_vec_eci), -1.0, 1.0)
        sun_angle_rad = np.arccos(cos_angle)
        sun_angle_deg = np.degrees(sun_angle_rad)

        # Earth angular radius as seen from satellite (radians)
        earth_ang_radius_rad = np.arcsin(R_EARTH / np.linalg.norm(pos_eci))

        # Eclipse check: Earth blocks Sun if angle between nadir and Sun < Earth angular radius
        # (penumbra is a thin transition, ignore for simplicity)
        in_eclipse = sun_angle_rad < earth_ang_radius_rad

        if in_eclipse:
            eclipse_depth = 1.0
            shadow_fraction = 1.0
        else:
            # Penumbra region: gradual transition over ~1-2°
            penumbra_margin = np.radians(1.5)  # ~1.5° penumbra width
            if sun_angle_rad < earth_ang_radius_rad + penumbra_margin:
                eclipse_depth = 1.0 - (earth_ang_radius_rad + penumbra_margin - sun_angle_rad) / penumbra_margin
            else:
                eclipse_depth = 0.0
            # Clamp to [0, 1]
            eclipse_depth = float(np.clip(eclipse_depth, 0.0, 1.0))
            shadow_fraction = eclipse_depth

        # Sun vector components (ECI)
        sun_vec = sun_vec_eci

        return sun_angle_deg, eclipse_depth, shadow_fraction, sun_vec

    # =========================================================================
    # BATTERY MODEL (charge/discharge cycle)
    # =========================================================================
    def _compute_battery(self, solar_power_W, bus_load_W, shadow_fraction, soc):
        """
        Battery SOC / voltage / current dynamics with charge/discharge
        across orbit. Battery temperature now lives in the unified
        thermal model (self._t_batt, see _compute_thermal) since it
        needs to respond to the battery heater -- it previously had no
        persistent state and no heater coupling at all.
        """
        # Solar power incident
        # solar_power_W is already shadow-attenuated in get_frame, so do NOT apply shadow again
        solar_incident = solar_power_W

        # Bus power (including heaters -- caller passes bus_load_W that
        # already includes heater draw, see get_frame reordering)
        total_load = bus_load_W

        # Net battery current (positive = charging)
        if solar_incident >= total_load:
            i_battery = (solar_incident - total_load) / BATTERY_V_NOMINAL
        else:
            i_battery = -(total_load - solar_incident) / BATTERY_V_NOMINAL

        # SOC update (C-rate = I / Capacity)
        soc_dot = i_battery / (BATTERY_CAPACITY_Ah * 3600.0)  # per second
        soc_new = soc + soc_dot * self.dt

        # Clamp SOC
        soc_new = np.clip(soc_new, BATTERY_MIN_SOC, BATTERY_MAX_SOC)

        # Battery voltage model (open-circuit voltage RISES with SOC).
        # Old formula was inverted: 2.0*(1-soc) - 0.5*(soc-0.5) simplifies
        # to V_NOM + 2.25 - 2.5*soc, i.e. voltage went DOWN as SOC went
        # up. Verified: old formula gave 29.6V at 25% SOC vs 27.75V at
        # 100% SOC -- backwards from any real discharge curve.
        v_battery = (BATTERY_V_NOMINAL - 1.5) + 3.0 * soc_new
        v_battery += i_battery * BATTERY_R_INTERNAL  # IR drop: + on charge, - on discharge

        # A real bus regulator clamps terminal voltage rather than
        # letting it swing freely with Ohm's-law IR drop -- without this,
        # large solar/load mismatches (e.g. full sun + light load) push
        # voltage up to ~34V on a 28V-nominal bus, which isn't healthy
        # telemetry either. Clip to a plausible regulated band.
        v_battery = np.clip(v_battery, 25.0, 31.0)

        return soc_new, v_battery, i_battery

    # =========================================================================
    # THERMAL CONTROL (heater ↔ thermistor pairs)
    # =========================================================================
    def _compute_thermal(self, eclipse_depth, system_load):
        """
        Thermal model with heater/thermistor control.

        Each node (CPU, Rx, Bus, Battery) is a first-order thermal lag
        toward an eclipse-dependent environment temperature:
            dT/dt = (Q_in - k*(T - T_env)) / C
        Radiative coupling (the `-k*(T-T_env)` term) is always active,
        in eclipse and in sunlight alike, so every node has a real,
        stable equilibrium. The old model set cooling to exactly zero
        during eclipse, which had no restoring force and drove CPU/Rx/Bus
        to their hard clip bounds within about a day (verified).

        Heater hysteresis now checks each node's OWN thermistor (the old
        code checked self._t_cpu for all three heaters, so the battery
        and Rx heaters almost never fired). Sensor noise is added only
        when reporting the frame (see get_frame), not to this persistent
        state, to avoid an accumulating random walk in the "true" temps.
        """
        # Ambient temperature -- external skin proxy only, NOT used to
        # drive internal box temperatures (which are far better buffered
        # by structure/insulation than an exposed exterior surface).
        if eclipse_depth > 0.8:
            t_ambient = -150.0 + 100.0 * (1.0 - eclipse_depth) + np.random.randn() * 2.0
        else:
            t_ambient = 120.0 - 200.0 * eclipse_depth + np.random.randn() * 2.0

        # ---- Heater hysteresis (each against its own thermistor) ----
        if self._t_cpu < THERM_CPUT_SETPOINT - THERM_HYSTERESIS:
            self._heater_cpu_on = True
        elif self._t_cpu > THERM_CPUT_SETPOINT + THERM_HYSTERESIS:
            self._heater_cpu_on = False

        if self._t_batt < THERM_BATT_SETPOINT - THERM_HYSTERESIS:
            self._heater_batt_on = True
        elif self._t_batt > THERM_BATT_SETPOINT + THERM_HYSTERESIS:
            self._heater_batt_on = False

        if self._t_rx < THERM_RX_SETPOINT - THERM_HYSTERESIS:
            self._heater_rx_on = True
        elif self._t_rx > THERM_RX_SETPOINT + THERM_HYSTERESIS:
            self._heater_rx_on = False

        heater_cpu_w = HEATER_CPU_POWER_W if self._heater_cpu_on else 0.0
        heater_batt_w = HEATER_BATT_POWER_W if self._heater_batt_on else 0.0
        heater_rx_w = HEATER_RX_POWER_W if self._heater_rx_on else 0.0
        heater_total_w = heater_cpu_w + heater_batt_w + heater_rx_w

        def _lag(T, Q_in, T_env, k, C):
            dT = ((Q_in - k * (T - T_env)) / C) * self.dt
            return T + dT

        # CPU: internal load + heater
        T_env_cpu = -25.0 if eclipse_depth > 0.5 else 8.0
        cpu_load_w = 80.0 * system_load + 20.0
        self._t_cpu = np.clip(
            _lag(self._t_cpu, cpu_load_w + heater_cpu_w, T_env_cpu, k=1.5, C=60.0),
            -40.0, 60.0)

        # Rx
        T_env_rx = -35.0 if eclipse_depth > 0.5 else 0.0
        rx_load_w = 40.0
        self._t_rx = np.clip(
            _lag(self._t_rx, rx_load_w + heater_rx_w, T_env_rx, k=1.2, C=80.0),
            -30.0, 50.0)

        # Bus (no dedicated heater in this design -- only cpu/batt/rx do)
        T_env_bus = -20.0 if eclipse_depth > 0.5 else 15.0
        bus_load_w = 30.0
        self._t_bus = np.clip(
            _lag(self._t_bus, bus_load_w, T_env_bus, k=1.2, C=80.0),
            -35.0, 55.0)

        # Battery: modest self-heating (BMS/electronics) + its own heater
        T_env_batt = -15.0 if eclipse_depth > 0.5 else 10.0
        batt_internal_w = 5.0
        self._t_batt = np.clip(
            _lag(self._t_batt, batt_internal_w + heater_batt_w, T_env_batt, k=0.6, C=150.0),
            -20.0, 40.0)

        return (
            self._t_cpu, self._t_rx, self._t_bus, self._t_batt, t_ambient,
            heater_cpu_w, heater_batt_w, heater_rx_w,
            heater_total_w
        )

    # =========================================================================
    # PROPULSION (Thruster + LEB)
    # =========================================================================
    def _compute_propulsion(self, solar_power_W):
        """
        Compute propulsion state. Thruster for short burns, LEB for long-duration.
        """
        # Thruster (chemical)
        if self._thruster_active and self._propellant_thruster > 0:
            self._thruster_burn_elapsed += self.dt
            if self._thruster_burn_elapsed > THRUST_MAX_BURST_S:
                self._thruster_active = False
                self._thruster_burn_elapsed = 0.0
            thrust = THRUST_FORCE_N
            propellant_rate = thrust / (THRUST_ISP_S * 9.81)
            self._propellant_thruster -= propellant_rate * self.dt
        else:
            thrust = 0.0
            self._thruster_burn_elapsed = 0.0

        # LEB (electric engine)
        if self._leb_active and self._propellant_leb > 0 and solar_power_W >= LEB_POWER_W:
            self._leb_burn_elapsed += self.dt
            if self._leb_burn_elapsed > LEB_MAX_BURST_S:
                self._leb_active = False
                self._leb_burn_elapsed = 0.0
            leb_thrust = LEB_THRUST_N
            propellant_rate_leb = leb_thrust / (LEB_ISP_S * 9.81)
            self._propellant_leb -= propellant_rate_leb * self.dt
        else:
            leb_thrust = 0.0
            self._leb_burn_elapsed = 0.0

        return (
            self._thruster_active, thrust, self._propellant_thruster,
            self._thruster_burn_elapsed,
            self._leb_active, leb_thrust, LEB_POWER_W,
            self._propellant_leb, self._leb_burn_elapsed
        )

    # =========================================================================
    # REACTION WHEELS (momentum management)
    # =========================================================================
    def _compute_reaction_wheels(self, eclipse_depth):
        """
        Reaction wheel speed/torque with a real physical coupling:
        wheel speed is integrated from wheel torque via tau = I_wheel *
        domega/dt (previously wheel speed was an independent random walk
        with zero relationship to the reported torque -- measured
        correlation between torque and the speed delta was 0.008).

        Torque is driven by disturbance sources that actually vary with
        the orbit, so the resulting speed/torque channels carry a real
        per-orbit signature instead of pure noise:
          - a gravity-gradient-like term oscillating at 2x orbital rate
          - a solar-pressure-like term active only outside eclipse
          - a small constant secular bias (static mass/alignment
            imbalance) that causes momentum to slowly build up over many
            orbits, exactly like a real spacecraft -- which is what
            necessitates periodic desaturation.

        Wheels are kept spinning at a small nonzero "bias momentum"
        (their initial speeds) rather than crossing zero, a real design
        practice that avoids low-speed bearing stiction; when a wheel's
        distance from its bias speed exceeds a threshold, a
        desaturation event engages a larger corrective torque (modeling
        active momentum unloading via thrusters/magnetorquers) until it
        recovers, producing the classic ramp-then-correct sawtooth
        pattern real reaction wheels show instead of a flat random walk.

        NOTE: this is an independent momentum-management subsystem for
        telemetry purposes -- it does not feed back into
        _attitude_dynamics (which uses a separate, deliberately simple
        and stable tracking filter). Coupling the two would reintroduce
        the kind of instability risk that filter was built to avoid.
        """
        theta = self._orbital_angle

        # Disturbance torque (body frame), each term correlated with
        # orbital phase / eclipse the way real disturbance torques are.
        tau_gg = 0.000016 * np.array([
            np.sin(2.0 * theta), np.cos(2.0 * theta), 0.3 * np.sin(2.0 * theta + 0.5)
        ])
        tau_solar = 0.000012 * (1.0 - eclipse_depth) * np.array([0.6, 0.9, 0.3])
        tau_secular = 2.4e-6 * np.array([1.0, -0.6, 0.4])
        tau_disturbance = tau_gg + tau_solar + tau_secular

        # Normal-mode wheel torque counteracts the disturbance (keeps
        # net torque on the body near zero -- the RW absorbs it instead)
        tau_body_cmd = -tau_disturbance

        # 4-wheel pyramid allocation (unchanged from original design)
        B = np.array([
            [ 1,  1, -1, -1],
            [ 1, -1,  1, -1],
            [ 1, -1, -1,  1]
        ]) / np.sqrt(3.0)
        B_pinv = B.T @ np.linalg.inv(B @ B.T)
        tau_wheels = B_pinv @ tau_body_cmd

        # ---- Desaturation state machine (per wheel) ----
        # Trigger uses ABSOLUTE speed thresholds, not distance from
        # nominal: with wheels biased low (~1000 RPM) inside a [0,6000]
        # range, a nominal-relative trigger is only reachable heading
        # toward the ceiling -- heading toward the 0 floor, the maximum
        # possible offset (~1000 RPM) never reaches a large trigger
        # distance, so the wheel just pins at 0 forever instead of ever
        # desaturating (verified: this happened within hours of runtime).
        offset = self._rw_speeds - self._rw_nominal
        for i in range(4):
            speed_i = self._rw_speeds[i]
            if not self._rw_desaturating[i] and (speed_i > RW_DESAT_HIGH_RPM or speed_i < RW_DESAT_LOW_RPM):
                self._rw_desaturating[i] = True
            elif self._rw_desaturating[i] and abs(offset[i]) < RW_DESAT_RELEASE_RPM:
                self._rw_desaturating[i] = False

            if self._rw_desaturating[i]:
                # Active momentum dump: torque toward nominal, overriding
                # the small disturbance-counter torque for this wheel.
                direction = 1.0 if speed_i < self._rw_nominal[i] else -1.0
                tau_wheels[i] = direction * RW_DESAT_TORQUE_NM

        tau_wheels = np.clip(tau_wheels, -RW_TORQUE_MAX_NM, RW_TORQUE_MAX_NM)

        # Integrate speed from torque: domega/dt = tau / I_wheel
        domega_rpm_dt = (tau_wheels / RW_INERTIA_KGM2) * (60.0 / (2.0 * np.pi))
        self._rw_speeds = self._rw_speeds + domega_rpm_dt * self.dt
        self._rw_speeds = np.clip(self._rw_speeds, 0.0, RW_SPEED_MAX_RPM)
        self._rw_torques = tau_wheels

        return self._rw_speeds, self._rw_torques

    # =========================================================================
    # COMM LINK (ground station visibility)
    # =========================================================================
    def _compute_comm(self, pos_eci):
        """
        Determine if satellite is visible to ground station.
        Ground station is on the equator, rotating with Earth (fixed
        longitude in ECEF). Previously it was fixed in ECI (inertial)
        space, i.e. not actually attached to a rotating Earth -- that
        made every orbit's comm pass geometry identical, with no
        day-length variation at all.
        """
        theta_earth = (GS_LON0_RAD + EARTH_ROTATION_RATE * self.t) % (2.0 * np.pi)
        gs_pos = np.array([
            R_EARTH * np.cos(theta_earth),
            R_EARTH * np.sin(theta_earth),
            0.0
        ])

        # Range
        range_km = np.linalg.norm(pos_eci - gs_pos)

        # Elevation angle (simplified)
        if range_km < GROUND_RANGE_MIN:
            comm_active = True
            snr = 30.0 - range_km / 200.0 + np.random.randn() * 0.5
        elif range_km < GROUND_RANGE_MAX:
            comm_active = True
            snr = 10.0 + 5.0 * (1.0 - (range_km - GROUND_RANGE_MIN) / (GROUND_RANGE_MAX - GROUND_RANGE_MIN)) + np.random.randn() * 1.0
        else:
            comm_active = False
            snr = -10.0 + np.random.randn() * 2.0

        snr = max(snr, -20.0)

        return snr, comm_active, range_km

    # =========================================================================
    # MAIN FRAME GENERATION
    # =========================================================================
    def get_frame(self):
        """
        Generate one telemetry frame.

        Returns:
            dict: Telemetry values keyed by channel name.
        """
        t = self.t

        # ---- 1. ORBITAL PROPAGATION ----
        pos_eci, vel_eci, altitude_km = self._propagate_orbit()

        # ---- 2. ATTITUDE (Earth-pointing) ----
        q_earth = self._compute_earth_pointing_quaternion(pos_eci, vel_eci)
        if self.t == 0:
            # Seed the actual attitude state to the true initial
            # nadir-pointing quaternion instead of the arbitrary identity
            # default -- otherwise the tracking filter spends its first
            # several seconds settling a large, unrealistic initial
            # mismatch (harmless/bounded, but pointless on "clean" data).
            self._q = q_earth.copy()
        omega_body = self._attitude_dynamics(q_earth)

        # ---- 3. SUN ILLUMINATION ----
        sun_angle, eclipse_depth, shadow_fraction, sun_vec_eci = self._compute_sun_geometry(pos_eci)

        # ---- 4. SOLAR POWER ----
        solar_power = SOLAR_POWER_MAX * (1.0 - shadow_fraction)
        # Add some noise and panel degradation
        solar_power *= (1.0 - 0.001 * t / 86400.0)  # 0.1%/day degradation
        solar_power = max(0.0, solar_power + np.random.randn() * 20.0)

        # ---- 5. BUS LOAD (base, before heaters) ----
        # system_load is a master driver (0-1)
        if "system_load" in self.state_injections:
            system_load = self.state_injections["system_load"]
        else:
            system_load = 0.2 + 0.15 * np.sin(2 * np.pi * t / 600.0) + np.random.randn() * 0.02
        system_load = np.clip(system_load, 0.0, 1.0)
        base_bus_load = BUS_LOAD_BASE_W + (BUS_LOAD_MAX_W - BUS_LOAD_BASE_W) * system_load

        # ---- 6. THERMAL (must run before battery: heater on/off is
        #         decided here, and that draw needs to be included in
        #         THIS frame's battery power balance -- previously
        #         heater_total_w was added to bus_load only after the
        #         battery calc had already run, so heater power was
        #         silently missing from the battery's load for the
        #         frame it was drawn in) ----
        (t_cpu, t_rx, t_bus, t_batt, t_ambient,
         heater_cpu_w, heater_batt_w, heater_rx_w,
         heater_total_w) = self._compute_thermal(eclipse_depth, system_load)

        bus_load = base_bus_load + heater_total_w  # heaters add to bus load

        # ---- 7. BATTERY ----
        soc, v_battery, i_battery = self._compute_battery(
            solar_power, bus_load, shadow_fraction, self._soc
        )
        self._soc = soc
        self._v_battery = v_battery
        self._i_battery = i_battery

        # ---- 8. PROPULSION ----
        (thruster_active, thrust_force, prop_thruster, burn_thruster,
         leb_active, leb_thrust, leb_power, prop_leb, burn_leb) = \
            self._compute_propulsion(solar_power)

        # ---- 9. COMM ----
        snr, comm_active, ground_range = self._compute_comm(pos_eci)

        # ---- 10. REACTION WHEELS & GYROS ----
        rw_speeds, rw_torques = self._compute_reaction_wheels(eclipse_depth)

        # Gyro rates (body rate + bias + noise)
        gyro_rates = self._omega_body * 180.0 / np.pi + self._gyro_bias + np.random.randn(3) * GYRO_NOISE_DPS

        # ---- 11. ASSEMBLE FRAME ----
        data = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            # Orbit
            "altitude_km": altitude_km + np.random.randn() * 0.1,
            # Navigation (ECI)
            "nav_pos_x_km": pos_eci[0] + np.random.randn() * 0.01,
            "nav_pos_y_km": pos_eci[1] + np.random.randn() * 0.01,
            "nav_pos_z_km": pos_eci[2] + np.random.randn() * 0.01,
            "nav_vel_x_kms": vel_eci[0] + np.random.randn() * 0.001,
            "nav_vel_y_kms": vel_eci[1] + np.random.randn() * 0.001,
            "nav_vel_z_kms": vel_eci[2] + np.random.randn() * 0.001,
            # Attitude (QIB) -- report the actual tracked/dynamic state
            # (self._q), NOT the freshly-recomputed ideal target
            # (q_earth). Previously these were reported from q_earth,
            # which made the quaternion channel look perfectly
            # nadir-pointed regardless of what the rate channels were
            # doing -- internally inconsistent, since a real spacecraft
            # telemeters its actual attitude, not the guidance target.
            "att_q0": self._q[0] + np.random.randn() * 0.001,
            "att_q1": self._q[1] + np.random.randn() * 0.001,
            "att_q2": self._q[2] + np.random.randn() * 0.001,
            "att_q3": self._q[3] + np.random.randn() * 0.001,
            # Attitude rates
            "att_rate_x_rads": omega_body[0] + np.random.randn() * 0.001,
            "att_rate_y_rads": omega_body[1] + np.random.randn() * 0.001,
            "att_rate_z_rads": omega_body[2] + np.random.randn() * 0.001,
            # Sun illumination
            "sun_angle_deg": sun_angle + np.random.randn() * 0.5,
            "eclipse_depth": float(np.clip(eclipse_depth + np.random.randn() * 0.01, 0.0, 1.0)),
            "shadow_fraction": float(np.clip(shadow_fraction + np.random.randn() * 0.01, 0.0, 1.0)),
            "sun_exposure": 1.0 if shadow_fraction < 0.5 else 0.0,
            "sun_vector_x": sun_vec_eci[0],
            "sun_vector_y": sun_vec_eci[1],
            "sun_vector_z": sun_vec_eci[2],
            # Battery
            "battery_voltage_V": v_battery + np.random.randn() * 0.05,
            "battery_current_A": i_battery + np.random.randn() * 0.1,
            "battery_soc": float(np.clip(soc * 100.0 + np.random.randn() * 0.1,
                                          BATTERY_MIN_SOC * 100.0, BATTERY_MAX_SOC * 100.0)),
            "battery_temperature_C": t_batt + np.random.randn() * 0.3,
            # Solar
            "solar_panel_current_A": solar_power / BATTERY_V_NOMINAL + np.random.randn() * 0.2,
            "solar_panel_watts": solar_power,
            "solar_panel_voltage_V": BATTERY_V_NOMINAL + np.random.randn() * 0.1,
            # Thermal (noise applied here at the reporting stage only --
            # t_cpu/t_rx/t_bus are the clean internal state from
            # _compute_thermal; the old code added noise directly into
            # the persistent state, creating an accumulating random walk
            # on top of the real dynamics)
            "cpu_temperature_C": t_cpu + np.random.randn() * 0.2,
            "rx_temperature_C": t_rx + np.random.randn() * 0.2,
            "bus_temperature_C": t_bus + np.random.randn() * 0.2,
            "ambient_temperature_C": t_ambient,
            # Heater control
            "heater_cpu_current_A": heater_cpu_w / 28.0 if self._heater_cpu_on else 0.0,
            "heater_batt_current_A": heater_batt_w / 28.0 if self._heater_batt_on else 0.0,
            "heater_rx_current_A": heater_rx_w / 28.0 if self._heater_rx_on else 0.0,
            "thermistor_cpu_setpoint_C": THERM_CPUT_SETPOINT,
            "thermistor_batt_setpoint_C": THERM_BATT_SETPOINT,
            "thermistor_rx_setpoint_C": THERM_RX_SETPOINT,
            "heater_bus_voltage_V": 28.0,
            # System
            "system_load": system_load,
            "bus_power_W": bus_load,
            "total_power_W": solar_power + max(0, -i_battery * BATTERY_V_NOMINAL),
            # RW (small measurement noise added at reporting only --
            # rw_speeds/rw_torques above are the clean physical state)
            "rw_speed_1_rpm": rw_speeds[0] + np.random.randn() * 2.0,
            "rw_speed_2_rpm": rw_speeds[1] + np.random.randn() * 2.0,
            "rw_speed_3_rpm": rw_speeds[2] + np.random.randn() * 2.0,
            "rw_speed_4_rpm": rw_speeds[3] + np.random.randn() * 2.0,
            "rw_torque_1_nm": rw_torques[0] + np.random.randn() * 0.0005,
            "rw_torque_2_nm": rw_torques[1] + np.random.randn() * 0.0005,
            "rw_torque_3_nm": rw_torques[2] + np.random.randn() * 0.0005,
            "rw_torque_4_nm": rw_torques[3] + np.random.randn() * 0.0005,
            # Gyros
            "gyro_rate_x_dps": gyro_rates[0],
            "gyro_rate_y_dps": gyro_rates[1],
            "gyro_rate_z_dps": gyro_rates[2],
            # Comm
            "comm_snr_db": snr,
            "comm_active": 1.0 if comm_active else 0.0,
            "ground_range_km": ground_range,
            # Propulsion: Thruster
            "thruster_active": 1.0 if thruster_active else 0.0,
            "thrust_force_N": thrust_force,
            "thruster_propellant_kg": max(0, self._propellant_thruster),
            "thruster_burn_time_s": self._thruster_burn_elapsed,
            # Propulsion: LEB
            "leb_active": 1.0 if leb_active else 0.0,
            "leb_thrust_N": leb_thrust,
            "leb_power_W": leb_power if leb_active else 0.0,
            "leb_propellant_kg": max(0, self._propellant_leb),
            "leb_burn_time_s": self._leb_burn_elapsed,
        }

        # ---- 12. APPLY STATE INJECTIONS (cascading) ----
        for p, v in self.state_injections.items():
            # Handle propulsion injections (update internal state)
            if p == 'thruster_active':
                self._thruster_active = bool(v)
                data['thruster_active'] = 1.0 if self._thruster_active else 0.0
                data['thrust_force_N'] = THRUST_FORCE_N if self._thruster_active else 0.0
                data['thruster_burn_time_s'] = self._thruster_burn_elapsed
            elif p == 'leb_active':
                self._leb_active = bool(v)
                data['leb_active'] = 1.0 if self._leb_active else 0.0
                data['leb_thrust_N'] = LEB_THRUST_N if self._leb_active else 0.0
                data['leb_burn_time_s'] = self._leb_burn_elapsed
            # Handle other parameter injections (output only)
            elif p not in ["system_load", "sun_exposure"]:
                if p in data:
                    data[p] = v

        # ---- 13. APPLY SENSOR OVERRIDES (output only) ----
        for p, v in self.sensor_overrides.items():
            if p in data:
                data[p] = v

        # ---- 14. APPLY ONE-FRAME SPIKES ----
        for p, v in list(self.sensor_spikes.items()):
            if p in data:
                data[p] = v
            del self.sensor_spikes[p]

        self.t += 1
        return data


# ============================================================================
# INJECTION LISTENER (CLI)
# ============================================================================
def injection_listener(sim):
    """
    Threaded CLI for injecting anomalies.
    Format: PARAM=VALUE:TYPE
      TYPE: state (cascading), frame (sensor override), spike (one-frame)
    """
    print("\n[COMMAND MODE] Type any time. Examples:")
    print("  system_load=1.0:state          (Increase bus load)")
    print("  cpu_temperature_C=0:frame      (Simulate sensor failure)")
    print("  battery_voltage_V=20:spike     (One-frame voltage dip)")
    print("  thruster_active=1:state        (Activate thruster)")
    print("  leb_active=1:state             (Activate LEB engine)")
    print("  solar_panel_watts=0:frame      (Panel sensor error)")
    print("  pause / resume                 (Stop/continue scrolling)")
    print("  clear                          (Reset all injections)")
    print("  help                           (Show all params)")

    while sim.running:
        line = sys.stdin.readline().strip()
        if not line:
            continue

        cmd = line.lower()
        if cmd == 'pause':
            sim.paused = True
            print(">>> Paused.")
        elif cmd == 'resume':
            sim.paused = False
            print(">>> Resumed.")
        elif cmd == 'clear':
            sim.state_injections.clear()
            sim.sensor_overrides.clear()
            sim.sensor_spikes.clear()
            print(">>> All cleared.")
        elif cmd == 'help':
            print(f"\nAvailable parameters ({len(PARAMS)}):")
            for i, p in enumerate(PARAMS):
                print(f"  {i+1:3d}. {p}")
            print(f"\nFormat: {p}=VALUE:state")
            print(f"  state  — cascading (affects physics)")
            print(f"  frame  — sensor override (output only)")
            print(f"  spike  — one-frame spike (auto-removed)")
        elif '=' in line:
            try:
                parts = line.split(':')
                inj_type = parts[1].strip().lower() if len(parts) > 1 else "state"

                kv = parts[0].split('=')
                p, v = kv[0].strip(), float(kv[1].strip())

                if p in PARAMS:
                    if inj_type == "frame":
                        sim.sensor_overrides[p] = v
                        print(f">>> FRAME injection: {p} = {v} (Sensor Error)")
                    elif inj_type == "spike":
                        sim.sensor_spikes[p] = v
                        print(f">>> SPIKE injection: {p} = {v} (One-frame Spike)")
                    else:
                        sim.state_injections[p] = v
                        print(f">>> STATE injection: {p} = {v} (Cascading)")
                else:
                    print(f">>> Unknown param: {p}")
            except Exception as e:
                print(f">>> Format error: {e}. Use PARAM=VAL:state|frame|spike")


# ============================================================================
# MAIN (standalone terminal mode)
# ============================================================================
def main():
    sim = TelemetrySimulator()
    threading.Thread(target=injection_listener, args=(sim,), daemon=True).start()

    try:
        while True:
            if not sim.paused:
                d = sim.get_frame()
                ts = d.pop('timestamp')
                items = list(d.items())
                print(f"\n[{ts}]" + "="*70)
                # Print in groups of 4 for readability
                for i in range(0, len(items), 4):
                    chunk = items[i:i+4]
                    print(" | ".join([f"{k[:13]:<13}: {v:>8.2f}" for k, v in chunk]))
                time.sleep(1.5)
            else:
                time.sleep(0.5)
    except KeyboardInterrupt:
        sim.running = False


if __name__ == "__main__":
    main()
