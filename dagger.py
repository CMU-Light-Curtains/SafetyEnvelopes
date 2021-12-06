"""DAgger for imitation learning"""
from typing import Tuple, NoReturn

import gym
import torch
import numpy as np
from sacred import SETTINGS
from sacred.utils import apply_backspaces_and_linefeeds
import tqdm

from common.buffers import ImitationBuffer
from envs.env import Env
from policies.actors import Pi
from policies.actors.actor import Actor
from policies.actors.baseline import BaselineActor
import policies.actors.nn.networks
import eval.metrics
import utils

# sacred
SETTINGS.DISCOVER_SOURCES = 'sys'
SETTINGS.CAPTURE_MODE = 'sys'

ex = utils.Experiment("DAgger")
ex.captured_out_filter = apply_backspaces_and_linefeeds
ex.add_source_file("dagger.py")
ex.add_config("config/dagger/vanilla_res/all.yaml")

# variant configs
ex.add_named_config("vres_all",      "config/dagger/vanilla_res/all.yaml")
ex.add_named_config("vres_scene5",   "config/dagger/vanilla_res/scene5.yaml")
ex.add_named_config("bevnet_all",    "config/dagger/bevnet/all.yaml")
ex.add_named_config("bevnet_scene5", "config/dagger/bevnet/scene5.yaml")
ex.add_named_config("ersnet_all",    "config/dagger/ersnet/all.yaml")
ex.add_named_config("ersnet_scene5", "config/dagger/ersnet/scene5.yaml")
ex.add_named_config("cnn1d_all",     "config/dagger/cnn1d/all.yaml")
ex.add_named_config("cnn1d_scene5",  "config/dagger/cnn1d/scene5.yaml")


@ex.automain
def dagger(_config):
    train_env = Env(split=_config["splits"]["train"],
                    debug=False, cam=False, pc=False, preload=False, progress=False)
    valid_env = Env(split=_config["splits"]["valid"],
                    debug=False, cam=False, pc=False, preload=False, progress=False)
    if _config["splits"]["test"] is not None:
        test_env = Env(split=_config["splits"]["test"],
                       debug=False, cam=False, pc=False, preload=False, progress=False)

    b_actor = BaselineActor()

    nn_actor_class = utils.get_class("network", _config['actor'])
    l_actor = nn_actor_class(thetas=train_env.thetas,
                             base_policy=b_actor,
                             min_range=train_env.min_range,
                             max_range=train_env.max_range)

    feat_dim, act_dim = l_actor.feat_dim, len(train_env.thetas)
    feat_space = gym.spaces.Box(low=-20 * np.ones(feat_dim, dtype=np.float32),
                                high=20 * np.ones(feat_dim, dtype=np.float32))
    act_space  = gym.spaces.Box(low=-20 * np.ones(act_dim, dtype=np.float32),
                                high=20 * np.ones(act_dim, dtype=np.float32))

    def split_size(train_size: int) -> Tuple[int, int]:
        assert train_size >= 1
        valid_size = max(int(_config["valid_ratio"] * train_size), 1)
        return train_size, valid_size

    # train and valid buffers
    train_buffer_size, valid_buffer_size = split_size(_config["buffer_size"])
    train_buffer = ImitationBuffer(buffer_size=train_buffer_size, observation_space=feat_space, action_space=act_space)
    valid_buffer = ImitationBuffer(buffer_size=valid_buffer_size, observation_space=feat_space, action_space=act_space)
    utils.cprint(f"Buffer sizes | Train: {train_buffer_size} | Valid: {valid_buffer_size}", "red")

    train_ep_index_iter = train_env.train_ep_index_iterator()

    # optimizer
    if _config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(l_actor.network.parameters(), lr=_config["learning_rate"])
    elif _config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(l_actor.network.parameters(), lr=_config["learning_rate"], momentum=0.9)
    else:
        raise Exception("optimizer must be adam/sgd")

    evaluator_class = utils.get_class("metric", _config['metric'])
    evaluator = evaluator_class(min_range=train_env.min_range, max_range=train_env.max_range, thetas=train_env.thetas)

    def add_samples_to_buffer(epoch,
                              behavior_cloning: bool,
                              num_train_samples: int):
        """
        Args:
            behavior_cloning (bool): whether to use ground truth actions or the current policy's actions.
            num_train_samples (int): number of train samples to be added into the train buffer.
        """
        l_actor.network.eval()

        # create mask of which examples will be added to the train buffer and valid buffer
        num_train, num_valid = split_size(num_train_samples)
        train_mask = np.random.permutation(np.hstack([np.ones(num_train),
                                                      np.zeros(num_valid)]).astype(np.bool))

        num_remaining = num_train + num_valid
        progress = tqdm.tqdm(total=num_remaining)
        progress.set_description("Collecting rollouts")

        ep_lens = []

        # infinite loop of episodes
        while num_remaining > 0:
            ep_len = 0
            vid, start = next(train_ep_index_iter)

            # a single episode
            l_actor.reset()

            # initializing episode: initial action
            init_envelope = train_env.reset(vid, start)
            act, logp_a, l_info = l_actor.init_action(init_envelope)  # act: (C,)

            # done means that the episode will transition into a terminal state due to this action
            # the next line initializes done to False. An episode can only terminate when the actor has control.
            done = False

            if behavior_cloning:  # use expert demonstration instead of policy's action
                obs, gt_act, end, e_info = train_env.step(action=None)
            else:
                obs, gt_act, end, e_info = train_env.step(action=act)

            # cutoff means that the episode needs to be cutoff even if the terminal state wasn't reached
            cutoff = (ep_len >= _config["horizon"]) or end or (num_remaining == 0)

            # step through episode: add entries to buffer
            while (not cutoff) and (not done):
                # actor step (even if control == False)
                act, logp_a, control, l_info = l_actor.step(obs)  # act: (C,)

                # take environment step
                if behavior_cloning:  # use expert demonstration instead of policy's action
                    obs, gt_act, end, e_info = train_env.step(action=None)
                else:
                    obs, gt_act, end, e_info = train_env.step(action=act)

                if control:
                    # add to buffer
                    feat = l_info["features"]  # (*AF)

                    # action to imitate in imitation learning
                    il_act = e_info["ss_action"] if _config["self_supervised"] else gt_act

                    if train_mask[num_remaining-1]:
                        train_buffer.add(feat, act, il_act)
                    else:
                        valid_buffer.add(feat, act, il_act)
                    progress.update(1)
                    num_remaining -= 1
                    ep_len += 1

                    # done means that the episode will transition into a terminal state due to this action
                    if not behavior_cloning:  # in behavior cloning, demonstration is used, so done is always False
                        done = train_env.done(act, gt_act)

                # the next line means that the episode needs to be cutoff even if the terminal state wasn't reached
                cutoff = (ep_len >= _config["horizon"]) or end or (num_remaining == 0)

            ep_lens.append(ep_len)

        progress.close()

        # logging
        ep_lens = np.array(ep_lens)
        avg_ep_len = ep_lens[ep_lens > 0].mean()
        ex.log_scalar("avg_ep_len", avg_ep_len, epoch)

        return f"[ Avg episode length: {avg_ep_len:.2f} ]"

    def update_policy(epoch, num_batches) -> str:
        # split into train and val views
        batch_size = _config["batch_size"]

        if _config["reinitialize_network"]:
            # randomly re-initialize weights
            utils.cprint("Re-initializing network weights ...", color="red")
            for layer in l_actor.network.children():
                layer.reset_parameters()

        # --------------------------------------------------------------------------------------------------------------
        # train
        # --------------------------------------------------------------------------------------------------------------
        l_actor.network.train()
        inf_batch_iter = train_buffer.inf_batches(batch_size=batch_size)
        for i in range(num_batches):
            batch = next(inf_batch_iter)
            assert len(batch.features) == batch_size
            optimizer.zero_grad()
            feats = torch.from_numpy(batch.features)  # (B, *AF)
            if torch.cuda.is_available():
                feats = feats.cuda()
            pi: Pi = l_actor.forward(feats)  # batch_shape=(B, C) event_shape=(,)

            if _config["hard_example_mining"]:
                # the next three lines implement hard example mining.
                # the weight of each example is just its loss.
                losses = evaluator.loss(pi, batch.demonstrations, reduce=False)  # (B,)
                weights = losses.detach()  # (B,)
                loss = (losses * weights).mean()  # (,)
            else:
                loss = evaluator.loss(pi, batch.demonstrations, reduce=True)  # (,)
            loss.backward()
            optimizer.step()

        # --------------------------------------------------------------------------------------------------------------
        # eval
        # --------------------------------------------------------------------------------------------------------------
        l_actor.network.eval()

        with torch.no_grad():
            # train loss
            train_losses = utils.AverageMeter("Train loss", ":.4e")
            for batch in train_buffer.single_pass(batch_size=batch_size):
                feats = torch.from_numpy(batch.features)  # (B, *AF)
                if torch.cuda.is_available():
                    feats = feats.cuda()
                pi: Pi = l_actor.forward(feats)  # batch_shape=(B, C) event_shape=()
                loss = evaluator.loss(pi, batch.demonstrations)
                train_losses.update(loss.item(), n=len(batch.features))

            # valid loss
            valid_losses = utils.AverageMeter("Valid loss", ":.4e")
            for batch in valid_buffer.single_pass(batch_size=batch_size):
                feats = torch.from_numpy(batch.features)  # (B, *AF)
                if torch.cuda.is_available():
                    feats = feats.cuda()
                pi: Pi = l_actor.forward(feats)  # batch_shape=(B, C) event_shape=()
                loss = evaluator.loss(pi, batch.demonstrations)
                valid_losses.update(loss.item(), n=len(batch.features))

        # logging
        ex.log_scalar("train_loss", train_losses.avg, epoch)
        ex.log_scalar("valid_loss", valid_losses.avg, epoch)

        return f"[ Train loss: {train_losses.avg:.5f} ] [ Valid loss: {valid_losses.avg:.5f} ]"

    def evaluate(epoch):
        """
        This function evaluates 3 policies (learned policy, base policy, oracle policy on the validation and test sets.
        It prints the overall evaluation metrics, and logs them to sacred too.
        """
        l_actor.network.eval()
        evaluator.reset()

        @utils.timer.time_fn(name="evaluation")
        def evaluate_split(env, split):
            def evaluate_policy(actor: Actor,
                                name: str,
                                oracle: bool) -> NoReturn:

                for vid, start in env.eval_ep_index_iterator():
                    ep_len = 0

                    # a single episode
                    actor.reset()

                    # initializing episode: initial action
                    init_envelope = env.reset(vid, start)
                    p_act, logp_a, p_info = actor.init_action(init_envelope)  # act: (C,)

                    obs, gt_act, end, e_info = env.step(p_act)
                    cutoff = (ep_len >= _config["horizon"]) or end

                    while not cutoff:
                        # actor step (even if control == False)
                        p_act, logp_a, control, p_info = actor.step(obs)  # p_act: (C,)

                        # take environment step
                        obs, gt_act, end, e_info = env.step(p_act)

                        if control:
                            p_pi = p_info['pi']  # BS=(C,) ES=()
                            evaluator.add(name=name, pi=p_pi, gt_action=gt_act)

                            # oracle action
                            if oracle:
                                o_act: np.ndarray = utils.valid_curtain_close_to_frontier(env.plannerV1, gt_act)  # (C,)
                                o_pi = Pi(actions=torch.from_numpy(o_act).unsqueeze(-1))  # BS=(C,) ES=()
                                evaluator.add(name='o', pi=o_pi, gt_action=gt_act)

                            # next line increases ep_len since an action was taken when control was True
                            ep_len += 1

                        # the next line means that the episode needs to be cutoff
                        cutoff = (ep_len >= _config["horizon"]) or end

            evaluate_policy(l_actor, name='l', oracle=True)
            evaluate_policy(b_actor, name='b', oracle=False)

            l_metric, b_metric, o_metric = evaluator.metric('l'), evaluator.metric('b'), evaluator.metric('o')

            print(f"==========================================================================")
            print(f"Evaluation for Split={split} Epoch={epoch}")
            print(f"--------------------------------------------------------------------------")
            print(f"Learned  policy: " + str(l_metric))
            print(f"Baseline policy: " + str(b_metric))
            print(f"Oracle   policy: " + str(o_metric))
            print(f"==========================================================================")

            # log metrics
            for metric, suffix in zip([l_metric, b_metric, o_metric], ["l", "b", "o"]):
                ex.log_scalar(f"{split}_huber_{suffix}", metric.huber, epoch)

            return l_metric, b_metric, o_metric

        valid_l_metric, _, _ = evaluate_split(valid_env, split="valid")
        if _config["splits"]["test"] is not None:
            evaluate_split(test_env, split="test")

        # return accuracy on valid split
        return valid_l_metric.huber

    ####################################################################################################################
    # MAIN LOOP
    ####################################################################################################################
    if _config["eval_every"] > 0:
        evaluate(epoch=0)

    best_valid_error = float('inf')

    # behavior cloning
    for epoch in range(1, _config["epochs"] + 1):

        # behavior cloning
        if epoch == 1:
            utils.cprint(f"Epoch {epoch}: Behavior Cloning", "red")
            print_msg3 = add_samples_to_buffer(epoch,
                                               behavior_cloning=True,
                                               num_train_samples=_config["samples_for_bc"])
            print_msg1 = update_policy(epoch, num_batches=_config["batches_for_bc"])
        else:
            utils.cprint(f"Epoch {epoch}: DAgger", "green")
            print_msg3 = add_samples_to_buffer(epoch,
                                               behavior_cloning=False,
                                               num_train_samples=_config["samples_per_epoch"])
            print_msg1 = update_policy(epoch, num_batches=_config["batches_per_epoch"])

        print_msg2 = f"[ Train buffer size: {train_buffer.num_samples()} ] " + \
                     f"[ Valid buffer size: {valid_buffer.num_samples()} ]"

        utils.pprint([f"EPOCH {epoch}", print_msg1, print_msg2, print_msg3])

        if _config["eval_every"] > 0:
            if epoch % _config["eval_every"] == 0 or epoch == _config["epochs"] + 1:
                valid_error = evaluate(epoch=epoch)

                # save network weights
                if valid_error < best_valid_error:
                    best_valid_error = valid_error
                    print(f"Saving network weights for epoch {epoch} with error {valid_error:1.4f}%  ...")
                    torch.save(l_actor.network.state_dict(), "/tmp/weights.pth")
                    ex.add_artifact("/tmp/weights.pth", name="actor_weights")

    ####################################################################################################################
