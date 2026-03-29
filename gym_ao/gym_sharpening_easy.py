import numpy as np
import matplotlib.pyplot as plt
import hcipy as hp
import gymnasium as gym  # 导入 gymnasium
from gymnasium.spaces import Box

# Global variables
DIAMETER = 8  # meter
WAVELENGTH = 1e-6  # meter
RESOLUTION = 256  # pixels
F_NUMBER = 40
OVERSAMPLING = 3  # pixels
N_AIRY = 8
N_PHOTONS = 1e10
N_MODES = 15
N_ACT_ACROSS = 20
MODE_BASIS = 'zernike'
FILTERED = True  # easy mode
WF_RMS = 1.7
DT = 1
DECOR_TIME = 30
PL_IDX = -2.5


class Sharpening_AO_system_easy(gym.Env):  # 1. 继承 gym.Env
    def __init__(self):
        super().__init__()  # 2. 初始化父类
        self.avg_rewards = []
        self.diameter = DIAMETER
        self.wavelength = WAVELENGTH
        self.pupil_grid = hp.make_pupil_grid(RESOLUTION, self.diameter)
        self.focal_length = F_NUMBER * self.diameter
        self.focal_grid = hp.make_focal_grid(
            q=OVERSAMPLING, num_airy=N_AIRY,
            pupil_diameter=self.diameter, reference_wavelength=self.wavelength,
            f_number=F_NUMBER)

        self.propagator = hp.FraunhoferPropagator(
            self.pupil_grid, self.focal_grid,
            focal_length=self.focal_length)
        self.spatial_resolution = self.focal_length * self.wavelength \
                                  / self.diameter
        self.aperture = hp.evaluate_supersampled(hp.make_vlt_aperture(),
                                                 self.pupil_grid, 4)
        self.wf_in = hp.Wavefront(self.aperture, self.wavelength)
        self.wf_in.total_power = 1
        self.ref_image = self.get_image(self.wf_in, dt=1, ref=True, noiseless=True)
        self.Inorm = self.ref_image.sum()
        self.num_photons = N_PHOTONS
        self.Ipeak = self.get_image(self.wf_in, dt=1, noiseless=True).max()
        self.cent_pixel = np.argmax(self.ref_image)
        self.make_dm()

        # 保持与 easy 原始逻辑相近的 action 范围，但结构对齐
        self.action_space = Box(low=-0.3, high=0.3, shape=(self.num_modes,),
                                dtype=np.float32)
        self.observation_space = Box(low=0, high=1.,
                                     shape=self.focal_grid.shape,
                                     dtype=np.float32)

        if MODE_BASIS == 'zernike':
            self.modal_norm = np.sqrt(
                np.array([hp.noll_to_zernike(i)[0] for i in np.arange(2, N_MODES + 3)]) ** (PL_IDX))
        else:
            self.modal_norm = 1.

        self.iteration = 0
        self.episode = 0
        self.tot_rewards = []
        self.reward_range = (0, np.inf)
        self.wf_rms = WF_RMS
        # 绘图对象延迟到 render 中初始化，或者保持在类属性中
        self.fig = None
        self.axes = None

    def step(self, action):
        action = action * self.modal_norm
        self.deformable_mirror.actuators += action / (2 * np.pi) * self.wavelength
        field_in = self.wf_in.copy()
        self.update_aberration()
        field_in.electric_field *= np.exp(1j * self.abb)
        self.image = self.get_image(field_in, noiseless=True) / self.Ipeak
        self.tot_image += self.image
        self.strehl = self.image[self.cent_pixel]
        self.observation = self.image - self.ref_image / self.Ipeak
        self.strehls.append(self.strehl)
        self.reward = self.strehl
        self.ep_reward += self.reward

        self.terminated = False
        self.truncated = bool(self.reward < 0.01)  # Explicitly cast to bool
        self.iteration += 1

        # EXPLICIT CASTING: Strip HCIPy properties before handing to SB3
        obs_array = np.asarray(self.observation.shaped, dtype=np.float32)
        reward_float = float(self.reward)

        return obs_array, reward_float, self.terminated, self.truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.reset_actuators()
        if hasattr(self, 'ep_reward'):
            avg_r = self.ep_reward / max(self.iteration, 1)
            self.avg_rewards.append(avg_r)
            self.tot_rewards.append(self.ep_reward)

        self.ep_reward = 0
        self.tot_image = 0
        self.strehls = []
        self.iteration = 0

        step_results = self.step(np.zeros(self.num_modes))

        # EXPLICIT CASTING
        observation = np.asarray(step_results[0], dtype=np.float32)

        self.episode += 1
        return observation, {}

    # def render(self):
    #     # 1. 初始化阶段：只在第一次调用时创建画布和对象
    #     if self.fig is None:
    #         plt.ion()  # 开启交互模式
    #         self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
    #
    #         # --- 初始化图1 (Image) ---
    #         self.axes[0, 0].axis('off')
    #         self.title1 = self.axes[0, 0].set_title('')
    #         # 注意：使用 .shaped 提取纯 NumPy 数组，配合原生 imshow 速度最快
    #         self.im1 = self.axes[0, 0].imshow(self.image.shaped, cmap='viridis', vmin=0)
    #         self.fig.colorbar(self.im1, ax=self.axes[0, 0])
    #
    #         # --- 初始化图2 (log10 Image) ---
    #         self.axes[0, 1].axis('off')
    #         self.title2 = self.axes[0, 1].set_title('log10 Image')
    #         self.im2 = self.axes[0, 1].imshow(np.log10(self.image.shaped + 1e-12), vmax=0, vmin=-4, cmap='inferno')
    #         self.fig.colorbar(self.im2, ax=self.axes[0, 1])
    #
    #         # --- 初始化图3 (DM Shape) ---
    #         self.axes[1, 0].axis('off')
    #         self.title3 = self.axes[1, 0].set_title('Deformable mirror shape')
    #         dm_phase = self.deformable_mirror.phase_for(self.wavelength) * self.aperture
    #         self.im3 = self.axes[1, 0].imshow(dm_phase.shaped, cmap='bwr', vmin=-1, vmax=1)
    #         self.fig.colorbar(self.im3, ax=self.axes[1, 0])
    #
    #         # --- 初始化图4 (Rewards 折线图) ---
    #         # 保持对折线对象 self.line 的引用
    #         self.line, = self.axes[1, 1].plot([], [], marker='o', color='black')
    #         self.axes[1, 1].set_ylabel('Episode reward')
    #         self.axes[1, 1].set_xlabel('Episode')
    #
    #         self.suptit = self.fig.suptitle('')
    #         self.fig.tight_layout()
    #         self.fig.show()
    #
    #     # 2. 数据更新阶段：极速刷新，不引起任何内存泄漏
    #     # 更新图1
    #     self.im1.set_data(self.image.shaped)
    #     self.title1.set_text(f'Image, Strehl: {self.strehl * 100:.2f}%')
    #
    #     # 更新图2
    #     self.im2.set_data(np.log10(self.image.shaped + 1e-12))
    #
    #     # 更新图3 (动态调整 colorbar 极值)
    #     dm_phase = self.deformable_mirror.phase_for(self.wavelength) * self.aperture
    #     vmax = np.max(np.abs(dm_phase.shaped)) + 1e-9
    #     self.im3.set_data(dm_phase.shaped)
    #     self.im3.set_clim(vmin=-vmax, vmax=vmax)  # 动态更新色阶范围
    #
    #     # 更新图4
    #     if self.episode > 1:
    #         self.line.set_data(np.arange(len(self.tot_rewards)), self.tot_rewards)
    #         self.axes[1, 1].relim()  # 重新计算坐标轴限制
    #         self.axes[1, 1].autoscale_view()  # 自动缩放视野适配新数据
    #
    #     self.suptit.set_text(f'Episode: {self.episode}, iteration: {self.iteration}')
    #
    #     # 3. 强制刷新 GUI 事件循环
    #     self.fig.canvas.draw()
    #     self.fig.canvas.flush_events()
    #     # 移除了 plt.pause()，因为它容易与主线程抢占资源导致假死

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)

    # 补充缺失的方法（拷贝自原文件）
    def get_random_aberration(self, rms):
        abb = hp.make_power_law_error(self.pupil_grid, 1., self.diameter, PL_IDX)
        if FILTERED:
            abb = self.influence_matrix.dot(self.P.dot(abb))
        abb = (abb - np.mean(abb[self.aperture > 0])) / \
              np.std(abb[self.aperture > 0]) * rms
        return abb

    def update_aberration(self):
        if self.iteration == 0:
            self.abb1 = self.get_random_aberration(WF_RMS)
            self.abb2 = self.get_random_aberration(WF_RMS)
        if (self.iteration * DT) % DECOR_TIME == 0:
            self.abb1 = self.abb2
            self.abb2 = self.get_random_aberration(WF_RMS)
        frac = ((self.iteration * DT) % DECOR_TIME) / DECOR_TIME
        self.abb = np.sqrt((1. - frac)) * self.abb1 + np.sqrt(frac) * self.abb2

    def get_image(self, wf, dt=1, ref=False, noiseless=True):
        im_out = self.get_focal_field(wf).intensity * dt
        if ref:
            return im_out
        else:
            if noiseless:
                return im_out * self.num_photons / self.Inorm
            else:
                return hp.Field(
                    np.random.poisson(im_out * self.num_photons / self.Inorm),
                    im_out.grid)

    def get_focal_field(self, wf, coro=False):
        if hasattr(self, 'deformable_mirror'):
            wf = self.deformable_mirror.forward(wf)
        if coro and hasattr(self, 'coronagraph'):
            wf = self.coronagraph.forward(wf)
        return self.propagator.forward(wf)

    def make_dm(self):
        if MODE_BASIS == 'actuators':
            num_act_across = N_ACT_ACROSS
            actuator_spacing = self.diameter / num_act_across
            influence_functions = hp.make_gaussian_influence_functions(
                self.pupil_grid, num_act_across, actuator_spacing)
            self.deformable_mirror = hp.DeformableMirror(influence_functions)
            self.num_modes = self.deformable_mirror.num_actuators
            self.influence_matrix = np.array(
                self.deformable_mirror.influence_functions.transformation_matrix.todense())
        elif MODE_BASIS == 'zernike':
            influence_functions = hp.make_zernike_basis(
                N_MODES + 1, self.diameter, self.pupil_grid, starting_mode=3)
            self.deformable_mirror = hp.DeformableMirror(influence_functions)
            self.num_modes = self.deformable_mirror.num_actuators
            self.influence_matrix = np.array(
                self.deformable_mirror.influence_functions.transformation_matrix)
        else:
            raise ValueError('Unknown modal basis {mode_basis}')
        if FILTERED:
            self.P = np.linalg.pinv(self.influence_matrix, rcond=1e-3)

    def reset_actuators(self):
        self.deformable_mirror.actuators = np.zeros(self.num_modes)


def run_sharpening():
    env = Sharpening_AO_system_easy()
    N_iter = 100
    N_episode = 10

    for episode in range(N_episode):
        # 接收新的返回值格式 (obs, info)
        o, info = env.reset()
        print('Episode:', env.episode)
        for i in range(N_iter):
            a = 0.1 * env.action_space.sample()
            # 接收 5 个返回值
            o, r, t, trunc, info = env.step(a)
            if t or trunc:
                break
            env.render()
    env.close()


if __name__ == '__main__':
    run_sharpening()