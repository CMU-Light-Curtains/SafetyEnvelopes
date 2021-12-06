from sacred import cli_option
import torch

from envs.env import Env
from eval.metrics.huber import Huber
from eval.metrics.standard import Standard
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


@cli_option('-S', '--split')
def split(args, run):
    run.info["split"] = str(args)


# `eval_unit` can be either
#   - "dataset" (default): combined evaluation over all videos in the dataset
#   - "video"            : evaluate each video separately
@cli_option('-E', '--eval_unit')
def eval_unit(args, run):
    run.info["eval_unit"] = str(args)


ex = utils.Experiment("eval", additional_cli_options=[load_weights_from_run, actor, horizon, split, eval_unit])


@ex.automain
def main():
    env = Env(split=ex.info["split"],
              debug=False,
              cam=False,
              pc=True,
              preload=False,
              progress=False)

    # BaselineActor
    if ("actor" not in ex.info) or (ex.info["actor"] == "BaselineActor"):
        actor: Actor = BaselineActor()
    # IdentityActor
    elif ex.info["actor"] == "IdentityActor":
        actor: Actor = IdentityActor(base_policy=BaselineActor())
    # VelocityActor
    elif ex.info["actor"] == "VelocityActor":
        actor: Actor = VelocityActor(thetas=env.thetas,
                                     base_policy=BaselineActor(),
                                     min_range=env.light_curtain.min_range,
                                     max_range=env.light_curtain.max_range)
    # NN Actor
    else:
        nn_actor_class = utils.get_class("network", ex.info["actor"])
        actor: Actor = nn_actor_class(thetas=env.thetas,
                                      base_policy=BaselineActor(),
                                      min_range=env.light_curtain.min_range,
                                      max_range=env.light_curtain.max_range)

        if "run_id" in ex.info:
            print(f"Loading weights from run {ex.info['run_id']} ...")
            file = utils.get_sacred_artifact_from_mongodb(run_id=ex.info["run_id"], name="actor_weights")
            state_dict = torch.load(file)
            actor.network.load_state_dict(state_dict)

        actor.network.eval()  # set network in eval mode

    horizon = ex.info["horizon"] if "horizon" in ex.info else float('inf')
    eval_unit = ex.info["eval_unit"] if "eval_unit" in ex.info else "dataset"
    assert eval_unit in ["dataset", "video"]

    # print cli options
    cli_option_pstr = \
        f"AGENT: {type(actor).__name__} | HORIZON: {horizon} | SPLIT: {ex.info['split']} | EVAL UNIT: {eval_unit}"
    if 'run_id' in ex.info:
        cli_option_pstr += f" | RUN: {ex.info['run_id']}"
    utils.pprint(cli_option_pstr)

    evaluator1 = Huber(min_range=env.min_range, max_range=env.max_range, thetas=env.thetas)
    evaluator2 = Standard(min_range=env.min_range, max_range=env.max_range, thetas=env.thetas)

    for vid_ in range(len(env.dataset)):
        # the next two lines reset the evaluator at the beginning of every video (for separate video evaluation)
        if eval_unit == "video":
            evaluator1.reset()
            evaluator2.reset()

        for vid, fid in env.single_video_index_iterator(vid_):
            ep_len = 0

            # a single episode
            actor.reset()

            # initializing episode: initial action
            init_envelope = env.reset(vid, fid)
            act, logp_a, p_info = actor.init_action(init_envelope)  # act: (C,)

            obs, gt_act, end, e_info = env.step(act)
            cutoff = (ep_len >= horizon) or end

            while not cutoff:
                # actor step (even if control == False)
                act, logp_a, control, p_info = actor.step(obs)  # act: (C,)

                # take environment step
                obs, gt_act, end, e_info = env.step(act)

                if control:
                    pi = p_info['pi']  # BS=(C,) ES=()
                    evaluator1.add(name='p', pi=pi, gt_action=gt_act)
                    evaluator2.add(name='p', pi=pi, gt_action=gt_act)

                    # the next line increases ep_len since an action was taken when control was True
                    ep_len += 1

                # the next line means that the episode needs to be cutoff
                cutoff = (ep_len >= horizon) or end

        # the entire video has been played. if separate video evaluation, print metrics for this video.
        if eval_unit == "video":
            utils.pprint(f"Evaluation results for video {vid_}: " +
                         f"{evaluator1.metric('p').str(total=True)} " +
                         f"{evaluator2.metric('p').str()}")

    if eval_unit == "dataset":
        utils.pprint(f"Evaluation results for entire split {ex.info['split']}: " +
                     f"{evaluator1.metric('p').str(total=False)} " +
                     f"{evaluator2.metric('p').str()}")
