from flask import Flask, jsonify, request
from flask_cors import CORS

from envs.sim import SimEnv
from eval.metrics.huber import Huber
from run.runner import Runner, ex

app = Flask("second")
CORS(app)


class SimRunner(Runner):
    def __init__(self):
        env = SimEnv(split="mini_train",
                     vis_app=app,
                     cam=False,
                     pc=True,
                     preload="horizon" not in ex.info,
                     progress="horizon" not in ex.info,
                     debug=False)
        super().__init__(env)

    def run_simulation(self, vid):
        evaluators = [Huber(min_range=self.env.min_range, max_range=self.env.max_range, thetas=self.env.thetas)]

        for vid, start in self.env.single_video_index_iterator(vid):
            self.run_episode(vid, start, evaluators)

    def stop_simulation(self):
        raise NotImplementedError


RUNNER = None


@app.route('/api/run_simulation', methods=['GET', 'POST'])
def run_simulation():
    global RUNNER
    instance = request.json
    response = {"status": "normal"}
    video_idx = instance["video_idx"]
    enable_int16 = instance["enable_int16"]

    RUNNER.run_simulation(video_idx)

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


@app.route('/api/stop_simulation', methods=['POST'])
def stop_simulation():
    global RUNNER
    instance = request.json
    response = {"status": "normal"}

    RUNNER.stop_simulation()

    response = jsonify(results=[response])
    response.headers['Access-Control-Allow-Headers'] = '*'
    return response


@ex.automain
def main(port=16666):
    global RUNNER
    RUNNER = SimRunner()

    app.run(host='127.0.0.1', threaded=True, port=port)
