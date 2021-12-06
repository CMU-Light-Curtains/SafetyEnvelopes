from envs.real import RealEnv
from eval.metrics.huber import Huber
from eval.metrics.standard import Standard
from run.runner import Runner, ex


class RealRunner(Runner):
    def __init__(self):
        env = RealEnv(debug=False)
        super().__init__(env)

    def run_simulation(self):
        evaluator1 = Huber(min_range=self.env.min_range, max_range=self.env.max_range, thetas=self.env.thetas)
        evaluator2 = Standard(min_range=self.env.min_range, max_range=self.env.max_range, thetas=self.env.thetas)
        evaluators = [evaluator1, evaluator2]

        self.run_episode(vid=None, start=None, evaluators=evaluators)


@ex.automain
def main():
    RUNNER = RealRunner()
    RUNNER.run_simulation()
