import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eigvals


rR    = 2.0      
alpha = 0.01     
bF    = 1/4      
B     = 1500.0    





K_R = 749.0      # carrying capacity for rabbits, change this. 










def dR(R, F):
    return rR * R * (1 - R/K_R) - rR * alpha * R * F

def dF(R, F):
    return (1/B) * R * F - bF * F

def jacobian(R, F):
    J11 = rR * (1 - 2*R/K_R) - rR * alpha * F
    J12 =      - rR * alpha * R
    J21 = (1/B) * F
    J22 = (1/B) * R - bF
    return np.array([[J11, J12],
                     [J21, J22]])

def classify(eqpt):
    J = jacobian(*eqpt)
    lam = eigvals(J)
    if np.all(lam.real < 0):
        return 'stable'
    if np.any(lam.real > 0):
        return 'unstable'
    return 'nonhyperbolic'

# find equilibria
eq_pts = [
    (0.0, 0.0),
    (K_R,  0.0),
    (B*bF, 0.0)
]
# interior if it exists
if K_R > B*bF:
    Rstar = B * bF          
    Fstar = (1 - Rstar/K_R) / alpha
    eq_pts.append((Rstar, Fstar))


R0, F0 = 375.0, 100*(1 - 375.0/K_R)    # print eigenvalues at the analytic interior point
print("Eigenvalues at (375, {:.3f}): {}".format(F0, eigvals(jacobian(R0, F0))))

# Phase‐plane plot
fig, ax = plt.subplots(figsize=(8,6))
P = np.linspace(0, 1.6*K_R, 25)
F = np.linspace(0, 0.8*K_R, 25)
Rmesh, Fmesh = np.meshgrid(P, F)

dRmesh = dR(Rmesh, Fmesh)
dFmesh = dF(Rmesh, Fmesh)
mag = np.hypot(dRmesh, dFmesh)
mag[mag==0] = 1.0
U, V = dRmesh/mag, dFmesh/mag
ax.quiver(Rmesh, Fmesh, U, V, color='gray', alpha=0.6, angles='xy', pivot='mid')

# nullclines
P_nc = np.linspace(0, 1.6*K_R, 400)
F_nc = (1 - P_nc/K_R)/alpha
ax.plot(P_nc, F_nc, 'b-', lw=2, label=r'$\dot R=0$')
ax.axvline(B*bF, color='r', lw=2, ls='--', label=r'$\dot F=0$')

# plot & classify equilibria
for R_e, F_e in eq_pts:
    cls = classify((R_e, F_e))
    marker = 'o' if cls=='stable' else 's' if cls=='unstable' else 'D'
    color  = 'green' if cls=='stable' else 'red' if cls=='unstable' else 'orange'
    ax.plot(R_e, F_e, marker=marker, color=color, ms=10,
            label=f'({R_e:.0f},{F_e:.1f}) {cls}')
    print((R_e, F_e), '→', cls)

ax.set_xlim(0, 1.6*K_R)
ax.set_ylim(0, 0.8*K_R)
ax.set_xlabel('R (rabbits)')
ax.set_ylabel('F (foxes)')
ax.set_title(rf'Phase Plane for $K_R={K_R}$')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True)
plt.tight_layout()
plt.show()