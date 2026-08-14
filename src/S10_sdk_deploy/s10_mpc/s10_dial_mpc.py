import numpy as np


class DialMpc:
    '''DiAL-MPC upper layer: DDP (iLQR) over the SRBD with continuous contact
    relaxation tau. Replaces the hard-coded distance-window swing schedule.

    State  x (12): body pos(3), vel(3), euler(3), omega(3)
    Control u (16): contact forces F(12) + contact relaxation tau(4) in [0,1]

    Cost: track vx/z/pitch/roll/yaw + contact entropy relu(tau(1-tau)) +
          force regularization.
    '''

    def __init__(self, mass=19.0, g=9.81, I=np.diag([0.15, 0.22, 0.30]),
                 wheelbase=0.456, track_half=0.18, mu=0.8, fz_max=180.0):
        self.m = mass
        self.g = g
        self.I = I
        self.wheelbase = wheelbase
        self.track_half = track_half
        self.mu = mu
        self.fz_max = fz_max
        # wheel attach positions in body frame (x forward, y left, z down?)
        self.attach = np.array([
            [wheelbase / 2, track_half, 0.0],
            [wheelbase / 2, -track_half, 0.0],
            [-wheelbase / 2, track_half, 0.0],
            [-wheelbase / 2, -track_half, 0.0],
        ])

    def dynamics(self, x, u, dt):
        p, v, th, w = x[0:3], x[3:6], x[6:9], x[9:12]
        F = u[0:12].reshape(4, 3)
        tau = np.clip(u[12:16], 0.0, 1.0)
        R = self._euler_R(th)
        # contact forces scaled by tau, in world frame (approximate: body frame)
        F_w = (R @ F.T).T
        F_eff = F_w * tau[:, None]
        a = np.sum(F_eff, axis=0) / self.m + np.array([0.0, 0.0, -self.g])
        # angular: sum r x F (r in body frame, F in world approx body frame)
        M = np.zeros(3)
        for i in range(4):
            M += np.cross(self.attach[i], F_eff[i])
        M += np.cross(w, self.I @ w)
        alpha = np.linalg.solve(self.I, M)
        x_next = np.concatenate([
            p + v * dt,
            v + a * dt,
            th + w * dt,
            w + alpha * dt,
        ])
        return x_next

    @staticmethod
    def _euler_R(th):
        roll, pitch, yaw = th
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        return Rz @ Ry @ Rx

    def cost(self, x, u, ref, w):
        p, v, th, w_ang = x[0:3], x[3:6], x[6:9], x[9:12]
        F = u[0:12].reshape(4, 3)
        tau = np.clip(u[12:16], 0.0, 1.0)
        c = 0.0
        c += w['vx'] * (v[0] - ref.get('vx', 0.0)) ** 2
        c += w['z'] * (p[2] - ref.get('z', 0.8)) ** 2
        c += w['pitch'] * (th[1] - ref.get('pitch', 0.0)) ** 2
        c += w['roll'] * (th[0] - ref.get('roll', 0.0)) ** 2
        c += w['yaw'] * (th[2] - ref.get('yaw', 0.0)) ** 2
        c += w['contact'] * np.sum(np.maximum(tau * (1.0 - tau), 0.0))
        c += w['force'] * np.sum(F ** 2)
        return c

    def solve(self, x0, ref, H=10, dt=0.05, w=None, n_iter=5, u0=None):
        if w is None:
            w = dict(vx=10.0, z=100.0, pitch=10.0, roll=20.0, yaw=5.0,
                     contact=0.1, force=1e-5)
        nx, nu = 12, 16
        if u0 is None:
            u0 = np.zeros((H, nu))
            u0[:, 12:16] = 1.0  # tau start at 1
            u0[:, 2::3] = self.m * self.g / 4.0  # F_z = mg/4 per wheel
        u_seq = u0.copy()
        x_seq = self._rollout(x0, u_seq, dt, H)
        for _ in range(n_iter):
            u_seq, x_seq = self._backward_forward(x0, u_seq, x_seq, ref, w, dt, H)
        return u_seq, x_seq

    def _rollout(self, x0, u_seq, dt, H):
        x_seq = np.zeros((H + 1, 12))
        x_seq[0] = x0
        for k in range(H):
            x_seq[k + 1] = self.dynamics(x_seq[k], u_seq[k], dt)
        return x_seq

    def _backward_forward(self, x0, u_seq, x_seq, ref, w, dt, H):
        nx, nu = 12, 16
        # terminal cost
        Vx = np.zeros(nx)
        Vxx = np.zeros((nx, nx))
        # backward pass
        k_gain = np.zeros((H, nu))
        K_gain = np.zeros((H, nu, nx))
        for k in reversed(range(H)):
            x, u = x_seq[k], u_seq[k]
            lx, lu, lxx, luu, lux = self._derivs(x, u, ref, w)
            fx, fu = self._dyn_derivs(x, u, dt)
            Qx = lx + fx.T @ Vx
            Qu = lu + fu.T @ Vx
            Qxx = lxx + fx.T @ Vxx @ fx
            Quu = luu + fu.T @ Vxx @ fu
            Qux = lux + fu.T @ Vxx @ fx
            Quu_reg = Quu + 1e-3 * np.eye(nu)
            k_gain[k] = -np.linalg.solve(Quu_reg, Qu)
            K_gain[k] = -np.linalg.solve(Quu_reg, Qux)
            Vx = Qx + K_gain[k].T @ Quu @ k_gain[k] + K_gain[k].T @ Qu + Qux.T @ k_gain[k]
            Vxx = Qxx + K_gain[k].T @ Quu @ K_gain[k] + K_gain[k].T @ Qux + Qux.T @ K_gain[k]
        # forward pass with line search
        alpha = 1.0
        for _ in range(5):
            x_new = np.zeros((H + 1, nx))
            u_new = np.zeros((H, nu))
            x_new[0] = x0
            for k in range(H):
                dx = x_new[k] - x_seq[k]
                u_new[k] = u_seq[k] + alpha * k_gain[k] + K_gain[k] @ dx
                u_new[k, 12:16] = np.clip(u_new[k, 12:16], 0.0, 1.0)
                x_new[k + 1] = self.dynamics(x_new[k], u_new[k], dt)
            if self._total_cost(x_new, u_new, ref, w) < self._total_cost(x_seq, u_seq, ref, w):
                return u_new, x_new
            alpha *= 0.5
        return u_seq, x_seq

    def _total_cost(self, x_seq, u_seq, ref, w):
        c = 0.0
        for k in range(len(u_seq)):
            c += self.cost(x_seq[k], u_seq[k], ref, w)
        _xt = x_seq[-1]
        c += 20.0 * ((_xt[2] - ref.get('z', 0.8)) ** 2 + (_xt[3] - ref.get('vx', 0.0)) ** 2)
        c += 20.0 * (_xt[6] ** 2 + _xt[7] ** 2)
        return c

    def _derivs(self, x, u, ref, w):
        # numeric derivatives of the scalar cost (simple and robust)
        nx, nu = 12, 16
        eps = 1e-4
        lx = np.zeros(nx)
        for i in range(nx):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            lx[i] = (self.cost(xp, u, ref, w) - self.cost(xm, u, ref, w)) / (2 * eps)
        lu = np.zeros(nu)
        for i in range(nu):
            up = u.copy(); up[i] += eps
            um = u.copy(); um[i] -= eps
            lu[i] = (self.cost(x, up, ref, w) - self.cost(x, um, ref, w)) / (2 * eps)
        lxx = np.zeros((nx, nx))
        for i in range(nx):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            lxx[i, :] = ((self._cost_grad(xp, u, ref, w) - self._cost_grad(xm, u, ref, w)) / (2 * eps))
        luu = np.zeros((nu, nu))
        for i in range(nu):
            up = u.copy(); up[i] += eps
            um = u.copy(); um[i] -= eps
            luu[i, :] = ((self._cost_ugrad(x, up, ref, w) - self._cost_ugrad(x, um, ref, w)) / (2 * eps))
        lux = np.zeros((nu, nx))
        for i in range(nu):
            up = u.copy(); up[i] += eps
            um = u.copy(); um[i] -= eps
            lux[i, :] = ((self._cost_grad(x, up, ref, w) - self._cost_grad(x, um, ref, w)) / (2 * eps))
        return lx, lu, lxx, luu, lux

    def _cost_grad(self, x, u, ref, w):
        nx = 12
        eps = 1e-4
        g = np.zeros(nx)
        for i in range(nx):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            g[i] = (self.cost(xp, u, ref, w) - self.cost(xm, u, ref, w)) / (2 * eps)
        return g

    def _cost_ugrad(self, x, u, ref, w):
        nu = 16
        eps = 1e-4
        g = np.zeros(nu)
        for i in range(nu):
            up = u.copy(); up[i] += eps
            um = u.copy(); um[i] -= eps
            g[i] = (self.cost(x, up, ref, w) - self.cost(x, um, ref, w)) / (2 * eps)
        return g

    def _dyn_derivs(self, x, u, dt):
        nx, nu = 12, 16
        eps = 1e-5
        fx = np.zeros((nx, nx))
        for i in range(nx):
            xp = x.copy(); xp[i] += eps
            xm = x.copy(); xm[i] -= eps
            fx[:, i] = (self.dynamics(xp, u, dt) - self.dynamics(xm, u, dt)) / (2 * eps)
        fu = np.zeros((nx, nu))
        for i in range(nu):
            up = u.copy(); up[i] += eps
            um = u.copy(); um[i] -= eps
            fu[:, i] = (self.dynamics(x, up, dt) - self.dynamics(x, um, dt)) / (2 * eps)
        return fx, fu


if __name__ == '__main__':
    mpc = DialMpc()
    x0 = np.zeros(12)
    x0[2] = 0.7  # body z
    x0[3] = 1.0  # vx
    ref = dict(vx=1.2, z=0.8, pitch=0.0, roll=0.0, yaw=0.0)
    u_seq, x_seq = mpc.solve(x0, ref, H=10, dt=0.05)
    print('tau final:', np.round(u_seq[-1, 12:16], 2))
    print('F_z final:', np.round(u_seq[-1, [2, 5, 8, 11]], 1))
    print('z trajectory:', np.round(x_seq[:, 2], 2))
