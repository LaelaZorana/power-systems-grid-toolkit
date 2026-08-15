"""Fault analysis: Zbus, symmetrical and unsymmetrical faults.

Zbus is the inverse of a Ybus that includes generator (source) impedances to
ground. With prefault voltage Vf (default 1.0 pu) the bolted three-phase
fault current at bus k is I_f = Vf / (Z_kk + Z_f) and the voltage at bus i
during the fault is V_i = Vf - Z_ik I_f.

Unsymmetrical faults use the sequence Thevenin impedances Z1, Z2, Z0 at the
faulted bus (Grainger and Stevenson, ch. 12):

    SLG  : I_a1 = Vf / (Z1 + Z2 + Z0 + 3 Zf),  I_a = 3 I_a1
    LL   : I_a1 = Vf / (Z1 + Z2 + Zf), I_a2 = -I_a1, I_b = -I_c = -j sqrt(3) I_a1
    DLG  : I_a1 = Vf / (Z1 + Z2 (Z0 + 3Zf) / (Z2 + Z0 + 3Zf))
           I_a2 = -I_a1 (Z0 + 3Zf) / (Z2 + Z0 + 3Zf), I_a0 = -I_a1 Z2 / (Z2 + Z0 + 3Zf)
"""

from __future__ import annotations

import numpy as np

A = np.exp(2j * np.pi / 3)
A_MAT = np.array([[1, 1, 1], [1, A**2, A], [1, A, A**2]], dtype=complex)  # 012 to abc


def build_zbus(n_bus, branches, source_impedances=None, shunts=None):
    """Zbus = inv(Ybus) with source impedances {bus: z} added to ground.

    'branches' entries need 'from', 'to', 'x' and optionally 'r'. Line charging
    and taps are ignored for the fault study, which is typical short circuit
    practice. Pass 'shunts' {bus: z_to_ground} to include them explicitly.
    """
    Y = np.zeros((n_bus, n_bus), dtype=complex)
    for br in branches:
        i, k = br["from"] - 1, br["to"] - 1
        y = 1.0 / complex(br.get("r", 0.0), br["x"])
        Y[i, i] += y
        Y[k, k] += y
        Y[i, k] -= y
        Y[k, i] -= y
    for src in (source_impedances or {}, shunts or {}):
        for bus, z in src.items():
            Y[bus - 1, bus - 1] += 1.0 / z
    return np.linalg.inv(Y)


def _check_bus(bus, n):
    if not 1 <= bus <= n:
        raise ValueError(f"bus {bus} out of range 1..{n}")


def three_phase_fault(zbus, bus, vf=1.0, zf=0.0):
    """Return (I_fault, V_bus_array) for a bolted (or Zf) three-phase fault."""
    _check_bus(bus, len(zbus))
    k = bus - 1
    i_f = vf / (zbus[k, k] + zf)
    v = vf - zbus[:, k] * i_f
    return i_f, v


def sequence_thevenin(z1bus, z2bus, z0bus, bus):
    _check_bus(bus, len(z1bus))
    k = bus - 1
    return z1bus[k, k], z2bus[k, k], z0bus[k, k]


def slg_fault(z1, z2, z0, vf=1.0, zf=0.0):
    """Single line to ground on phase a. Returns dict of sequence and phase currents."""
    ia1 = vf / (z1 + z2 + z0 + 3 * zf)
    i012 = np.array([ia1, ia1, ia1])
    iabc = A_MAT @ i012
    return dict(I012=i012, Iabc=iabc, If=iabc[0])


def ll_fault(z1, z2, vf=1.0, zf=0.0):
    """Line to line between phases b and c."""
    ia1 = vf / (z1 + z2 + zf)
    i012 = np.array([0.0, ia1, -ia1])
    iabc = A_MAT @ i012
    return dict(I012=i012, Iabc=iabc, If=iabc[1])


def dlg_fault(z1, z2, z0, vf=1.0, zf=0.0):
    """Double line to ground on phases b and c."""
    zpar = z2 * (z0 + 3 * zf) / (z2 + z0 + 3 * zf)
    ia1 = vf / (z1 + zpar)
    ia2 = -ia1 * (z0 + 3 * zf) / (z2 + z0 + 3 * zf)
    ia0 = -ia1 * z2 / (z2 + z0 + 3 * zf)
    i012 = np.array([ia0, ia1, ia2])
    iabc = A_MAT @ i012
    return dict(I012=i012, Iabc=iabc, If=iabc[1] + iabc[2])  # ground current 3 Ia0


def all_faults_at_bus(z1bus, z2bus, z0bus, bus, vf=1.0, zf=0.0):
    """Magnitudes of fault current for 3ph, SLG, LL and DLG at one bus."""
    z1, z2, z0 = sequence_thevenin(z1bus, z2bus, z0bus, bus)
    i3, _ = three_phase_fault(z1bus, bus, vf, zf)
    return {
        "3ph": abs(i3),
        "SLG": abs(slg_fault(z1, z2, z0, vf, zf)["If"]),
        "LL": abs(ll_fault(z1, z2, vf, zf)["If"]),
        "DLG": abs(dlg_fault(z1, z2, z0, vf, zf)["If"]),
    }
