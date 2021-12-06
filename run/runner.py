import torch
from sacred import cli_option

from envs.env import Env
from policies.actors.actor import Actor
from policies.actors.baseline import BaselineActor
from policies.actors.identity import IdentityActor
from policies.actors.velocity import VelocityActor
import policies.actors.nn.networks
import utils


@cli_option('-l', '--load_weights_from_run')
def load_weights_from_run(args, run):
    run.info["run_id"] = int(args)


@cli_option('-A', '--actor')
def actor(args, run):
    run.info["actor"] = str(args)


@cli_option('-H', '--horizon')
def horizon(args, run):
    run.info["horizon"] = int(args)


@cli_option('-E', '--eval', is_flag=True)
def eval_flag(args, run):
    run.info["eval"] = True


ex = utils.Experiment("run", additional_cli_options=[load_weights_from_run, actor, horizon, eval_flag])


class Runner:
    def __init__(self, env: Env):
        self.env = env

        # BaselineActor
        if ("actor" not in ex.info) or (ex.info["actor"] == "BaselineActor"):
            self.actor: Actor = BaselineActor(**self.env.baseline_config)
        # IdentityActor
        elif ex.info["actor"] == "IdentityActor":
            self.actor: Actor = IdentityActor(base_policy=BaselineActor(**self.env.baseline_config))
        # VelocityActor
        elif ex.info["actor"] == "VelocityActor":
            self.actor: Actor = VelocityActor(thetas=self.env.thetas,
                                              base_policy=BaselineActor(**self.env.baseline_config))
        # NN Actor
        else:
            nn_actor_class = utils.get_class("network", ex.info["actor"])
            self.actor: Actor = nn_actor_class(thetas=self.env.thetas,
                                               base_policy=BaselineActor(**self.env.baseline_config),
                                               min_range=self.env.min_range,
                                               max_range=self.env.max_range)

            if "run_id" in ex.info:
                print(f"Loading weights from run {ex.info['run_id']} ...")
                file = utils.get_sacred_artifact_from_mongodb(run_id=ex.info["run_id"], name="actor_weights")
                state_dict = torch.load(file)
                self.actor.network.load_state_dict(state_dict)

            self.actor.network.eval()  # set network in eval mode

        self.horizon = ex.info["horizon"] if "horizon" in ex.info else float('inf')
        self.eval = "eval" in ex.info

        # print cli options
        cli_option_pstr = f"AGENT: {type(self.actor).__name__} | HORIZON: {self.horizon}"
        if 'run_id' in ex.info:
            cli_option_pstr += f" | RUN: {ex.info['run_id']}"
        utils.pprint(cli_option_pstr)

    def run_episode(self, vid, start, evaluators):
        ep_len = 0

        # a single episode
        self.actor.reset()

        # initializing episode: initial action
        init_envelope = self.env.reset(vid, start)
        act, logp_a, p_info = self.actor.init_action(init_envelope)  # act: (C,)

        obs, end, e_info = self.env.step(act, score=0.0, get_gt=False)
        cutoff = (ep_len >= self.horizon) or end

        score = None  # score is only changed under control

        # simulation loop
        while not cutoff:
            # actor step (even if control == False)
            act, logp_a, control, p_info = self.actor.step(obs)  # act: (C,)

            # take environment step
            obs, end, e_info = self.env.step(act, score=score, get_gt=self.eval)

            if control:
                if self.eval:  # evaluation
                    pi = p_info['pi']  # BS=(C,)  ES=()
                    gt_act = e_info['gt_action']  # (C,)
                    for evaluator in evaluators:
                        score = evaluator.add(name='p', pi=pi, gt_action=gt_act)

                # the next line increases ep_len since an action was taken when control was True
                ep_len += 1
            cutoff = (ep_len >= self.horizon) or end
        
        # episode has ended: print current cumulative metrics every time this happens
        if self.eval:
            for evaluator in evaluators:
                utils.pprint(str(evaluator.metric('p')))
