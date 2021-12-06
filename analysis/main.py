import matplotlib.pyplot as plt
import numpy as np
import tqdm

from lc_planner.config import LCConfig
from lc_planner.planner import PlannerRT
from analysis.box import Box
from envs.env import Env
import utils
import pickle

ex = utils.Experiment("analysis")


class Analysis:
    def __init__(self):
        # use the planner that is used in RealEnv
        self.lc_config = LCConfig(config_file='config/lc/real.yaml',
                                  default_config_file='LC-PLANNER/apis/py/lc_planner/defaults.yaml')
        
        # jeep settings used in the original analysis
        self.min_range = 0.50  # on jeep
        self.max_range = 10.0  # on jeep
        self.lc_config.LASER_PARAMS["max_alpha"] = 1.5e+7  # on jeep

        ranges = Env.get_ranges(self.min_range, self.max_range, 300, 1)
        self.planner = PlannerRT(self.lc_config, ranges, self.lc_config.CAMERA_PARAMS["width"], version=2)
        self.boxes = None

    def compute_dp_hit_prob_for_all_boxes(self, threshold, r_sampling, debug=False):
        description = f"Computing hit probs for threshold={threshold}, r_sampling={r_sampling}"
        for box in tqdm.tqdm(self.boxes, desc=description):
            self.compute_dp_hit_prob_for_single_box(box, threshold=threshold, r_sampling=r_sampling, debug=debug)

    def compute_dp_hit_prob_for_single_box(self, box, threshold, r_sampling, debug=False):
        surface = box.surface()
        self.planner._planner.loadLineSurface(surface)
        self.planner._planner.computeVisibleRanges()

        if debug:
            visible_pts = self.planner._planner.visiblePoints()
            visible_pts = np.array(visible_pts)
            if len(visible_pts) > 0:
                self.planner.draw_boundary()
                plt.scatter(visible_pts[:, 0], visible_pts[:, 1], c='r')
            plt.show()

        self.planner._planner.computeLayoutIntensities()

        if debug:
            layout_intensities = self.planner._planner.layoutIntensities()
            layout_intensities = np.array(layout_intensities)  # (RAYS, RANGES_PER_RAY)
            assert not np.isnan(layout_intensities).any()

            layout = self.planner._planner.getLayoutForVis()
            layout = np.array(layout)  # (RAYS, RANGES_PER_RAY)
            locs = layout.ravel()  # (RAYS * RANGES_PER_RAY,)

            x = np.array([loc.x for loc in locs])
            z = np.array([loc.z for loc in locs])
            i = layout_intensities.ravel()

            # sort according to increasing intensities
            inds = np.argsort(i)
            x, z, i = x[inds], z[inds], i[inds]

            plt.scatter(visible_pts[:, 0], visible_pts[:, 1], c='r', s=3)
            plt.scatter(x, z, c=i, cmap='viridis', s=1, vmin=0.0, vmax=1.0)
            plt.title("Intensities for polygon", fontsize='x-large')
            plt.show()

        hit_prob = self.planner._planner.randomCurtainHitProb(threshold, r_sampling)
        box.hit_info = dict(prob=hit_prob, threshold=threshold, r_sampling=r_sampling)

    def compute_mc_hit_prob_for_single_box(self, box, threshold, r_sampling, debug=False):
        surface = box.surface()
        self.planner._planner.loadLineSurface(surface)
        self.planner._planner.computeVisibleRanges()

        if debug:
            visible_pts = self.planner._planner.visiblePoints()
            visible_pts = np.array(visible_pts)
            if len(visible_pts) > 0:
                self.planner.draw_boundary()
                plt.scatter(visible_pts[:, 0], visible_pts[:, 1], c='r')
            plt.show()

        self.planner._planner.computeLayoutIntensities()

        if debug:
            layout_intensities = self.planner._planner.layoutIntensities()
            layout_intensities = np.array(layout_intensities)  # (RAYS, RANGES_PER_RAY)
            assert not np.isnan(layout_intensities).any()

            layout = self.planner._planner.getLayoutForVis()
            layout = np.array(layout)  # (RAYS, RANGES_PER_RAY)
            locs = layout.ravel()  # (RAYS * RANGES_PER_RAY,)

            x = np.array([loc.x for loc in locs])
            z = np.array([loc.z for loc in locs])
            i = layout_intensities.ravel()

            # sort according to increasing intensities
            inds = np.argsort(i)
            x, z, i = x[inds], z[inds], i[inds]

            plt.scatter(visible_pts[:, 0], visible_pts[:, 1], c='r', s=3)
            plt.scatter(x, z, c=i, cmap='viridis', s=1, vmin=0.0, vmax=1.0)
            plt.title("Intensities for polygon", fontsize='x-large')
            plt.show()

        curtain = np.array(self.planner._planner.randomCurtainDiscrete(r_sampling))  # (C, 3)
        intensities = curtain[:, 2]  # (C,)
        hit = np.any(intensities > threshold)
        box.hit_info = dict(hit=hit, threshold=threshold, r_sampling=r_sampling)

    def render(self, box, title="", show=False):
        self.planner.draw_boundary()
        box.draw()
        plt.title(title, fontsize='xx-large')
        plt.gca().set_aspect(1)
        if show:
            plt.show()

    @staticmethod
    def load_kitti_sizes():
        data = np.loadtxt("analysis/kitti_sizes.txt", dtype=str)
        data = {row[0]: (float(row[1]), float(row[2])) for row in data}
        return data

    def plot_prob_vs_area(self, threshold, r_sampling, savepath=None):
        self.boxes = []
        for side_length in np.arange(0.2, 4, step=0.2):
            for rot in [0, 15, 30, 45]:
                box = Box(x=0, z=0.75 * self.max_range, w=side_length, h=side_length, rot=rot, translate='center')
                # self.render(box, show=True)
                # assert box.is_inside_range(self.max_range)
                self.boxes.append(box)

        self.compute_dp_hit_prob_for_all_boxes(threshold=threshold, r_sampling=r_sampling)

        xs = np.array([box.area() for box in self.boxes])
        ys = np.array([box.hit_info['prob'] for box in self.boxes])

        # average probs across shapes of a fixed area
        unique_xs = np.array(sorted(set(xs)))
        unique_ys = [np.array([y for x2, y in zip(xs, ys) if x1 == x2]).mean() for x1 in unique_xs]

        plt.scatter(unique_xs, unique_ys, c='r')
        plt.plot(unique_xs, unique_ys, c='r')

        plt.xlabel("Area of the object ($m^2$)", fontsize='xx-large')
        plt.ylabel("Single curtain detection probability", fontsize='xx-large')
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')

        threshold = self.boxes[0].hit_info['threshold']
        sampling = self.boxes[0].hit_info['r_sampling']
        title = f"Threshold={threshold}, Sampling={sampling}"
        if savepath is None:
            plt.title(title, fontsize='xx-large')
        plt.tight_layout()

        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, format='png', metadata=dict(title=title))
        plt.clf()

    def plot_prob_vs_curtains(self, threshold, r_sampling, savepath=None, debug=False):
        kitti_sizes = self.load_kitti_sizes()
        self.boxes = []
        for class_name, (w, h) in kitti_sizes.items():
            for rot in [0, 45, 90, 135]:
                box = Box(x=0, z=0.75 * self.max_range, w=w, h=h, rot=rot, translate='near_edge')
                box.class_name = class_name
                if debug:
                    self.render(box, title=class_name, show=True)
                self.boxes.append(box)

        self.compute_dp_hit_prob_for_all_boxes(threshold=threshold, r_sampling=r_sampling, debug=False)

        # aggregate by class
        cns = sorted(set([box.class_name for box in self.boxes]))
        cn_to_ps = {cn: np.array([box.hit_info['prob'] for box in self.boxes if box.class_name == cn]) for cn in cns}

        colors = plt.get_cmap('tab10').colors
        xs = np.arange(1, 11)  # number of curtain placements
        times = xs * (1000 / 60.0)  # 60 fps
        for i, (cn, ps) in enumerate(cn_to_ps.items()):
            ys = []
            for x in xs:
                # hit probability in x light curtain placements
                y = 1.0 - np.power(1.0 - ps, x)
                ys.append(y.mean())
            color = np.array(colors[i])
            plt.plot(times, ys, label=cn, c=color)
            plt.scatter(times, ys, c=np.atleast_2d(color))

        plt.legend(fontsize='xx-large')
        plt.xlabel('Time taken to place multiple curtains (ms)', fontsize='xx-large')
        plt.ylabel('Multi-curtain detection probability', fontsize='xx-large')
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')
        title = f"Threshold={threshold}, Sampling={r_sampling}"
        if savepath is None:
            plt.title(title, fontsize='xx-large')
        plt.tight_layout()

        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, format='png', metadata=dict(title=title))
        plt.clf()

    def monte_carlo_comparison(self, threshold, r_sampling, savepath=None, save=False, load=False):
        NUM_DP = 10
        NUM_MC = 20000

        box = Box(x=0, z=0.75 * self.max_range, w=2, h=2, rot=0, translate='center')  # a single box

        data_path = "analysis/plots/mc.npy"
        if load:
            utils.cprint(f"Loading from {data_path} ...", color="yellow")
            with open(data_path, 'rb') as f:
                data = pickle.load(f)

            dp_time = data["dp_time"]
            dp_prob = data["dp_prob"]
            mc_times = data["mc_times"]
            mc_probs = data["mc_probs"]
            mc_stddevs = data["mc_stddevs"]

        else:
            # dynamic programming
            for i in tqdm.trange(NUM_DP, desc="Running DP to compute hit_prob"):
                with utils.timer.time_as("dp"):
                    self.compute_dp_hit_prob_for_single_box(box, threshold=threshold, r_sampling=r_sampling, debug=False)

            dp_time = utils.timer.ttime["dp"] / utils.timer.titer["dp"]
            dp_prob = box.hit_info["prob"]

            # monte carlo estimation
            mc_hits = []
            for i in tqdm.trange(NUM_MC, desc="Running MC to compute hit_prob"):
                with utils.timer.time_as("mc"):
                    self.compute_mc_hit_prob_for_single_box(box, threshold=threshold, r_sampling=r_sampling, debug=False)
                    mc_hits.append(box.hit_info['hit'])

            mc_time = utils.timer.ttime["mc"] / utils.timer.titer["mc"]
            mc_hits = np.array(mc_hits, dtype=np.float32)

            mc_probs, mc_stddevs, mc_times = [], [], []
            mc_std = mc_hits.std()

            for num_hits in np.logspace(np.log10(510), np.log10(NUM_MC), 10):
                num_hits = int(num_hits)
                sample = np.random.choice(mc_hits, num_hits, replace=False)
                mean = sample.mean()
                stddev = 1.96 * mc_std / np.sqrt(num_hits)
                time = num_hits * mc_time

                mc_probs.append(mean)
                mc_stddevs.append(stddev)
                mc_times.append(time)

        if save:
            utils.cprint(f"Saving to {analysis/plots/mc.npy} ...", color="yellow")
            data = dict(dp_time=dp_time, dp_prob=dp_prob, mc_times=mc_times, mc_probs=mc_probs, mc_stddevs=mc_stddevs)
            with open(analysis/plots/mc.npy, 'wb') as f:
                pickle.dump(data, f)

        utils.cprint(f"dp_time: {dp_time}")

        # plot limits for logspace plots
        all_times = np.array(mc_times + [dp_time])
        x_min_log, x_max_log = np.log10(all_times.min()), np.log10(all_times.max())
        range_log = x_max_log - x_min_log
        # expand plot range by 5%
        x_min_log -= 0.05 * range_log
        x_max_log += 0.05 * range_log
        x_min, x_max = np.power(10, x_min_log), np.power(10, x_max_log)

        # plotting
        # plt.gca().axhline(y=dp_prob, c='r', zorder=-1)
        plt.plot([dp_time, x_max], [dp_prob, dp_prob], c='r')
        plt.scatter([dp_time], [dp_prob], color='r', s=50, label='Dynamic Programming')
        plt.errorbar(mc_times, mc_probs, yerr=mc_stddevs, color='b', capsize=6, fmt='o', label='Monte Carlo Sampling')
        plt.xscale('log')
        plt.xlim([x_min, x_max])
        plt.xlabel('Runtime of the algorithm (sec)', fontsize='xx-large')
        plt.ylabel('Single curtain det. prob. estimate', fontsize='xx-large')
        plt.legend(fontsize='x-large')
        plt.xticks(fontsize='x-large')
        plt.yticks(fontsize='x-large')
        title = f"Threshold={threshold}, Sampling={r_sampling}"
        if savepath is None:
            plt.title(title, fontsize='xx-large')
        plt.tight_layout()

        if savepath is None:
            plt.show()
        else:
            plt.savefig(savepath, format='png', metadata=dict(title=title))
        plt.clf()


@ex.automain
def main():
    analysis = Analysis()
    analysis.plot_prob_vs_area(threshold=0.5, r_sampling="linear", savepath="analysis/plots/prob_vs_area.png")
    analysis.plot_prob_vs_curtains(threshold=0.5, r_sampling="linear", savepath="analysis/plots/prob_vs_curtains.png")
    analysis.monte_carlo_comparison(threshold=0.5, r_sampling="linear",
                                    savepath="analysis/plots/monte_carlo.png", load=False, save=False)
