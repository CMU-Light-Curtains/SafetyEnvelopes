import torch
import torch.distributions as td
from typing import Optional


class Pi (td.Categorical):
    def __init__(self,
                 actions: torch.Tensor,
                 logits: Optional[torch.Tensor] = None,
                 probs: Optional[torch.Tensor] = None):
        """
        Args:
            actions (torch.Tensor, dtype=float32, shape=(*L, C, R): actions per camera ray.
            logits Optional[torch.Tensor, dtype=float32, shape=(*L, C, R)]: logits over actions for every camera ray.
            probs Optional[torch.Tensor, dtype=float32, shape=(*L, C, R)]: probs over actions for every camera ray.
                `probs' should sum to 1 over the last dimension.

        NOTE: the device (cpu/cuda) of actions and logits/probs should match.
        """
        if logits is None and probs is None:
            # single action per camera ray. all probs are 1.
            assert actions.shape[-1] == 1
            probs = torch.ones_like(actions)  # (*L, C, R=1)

        super().__init__(logits=logits, probs=probs)  # (td.Categorical batch_shape=(*L, C) event_shape=())

        # we support inputting actions with singleton batch dims, even if logits/probs are not singleton.
        # so we will explicitly expand it here.
        self.actions = actions.expand(self.probs.shape)

    def sample(self):
        """
        Returns:
            act (torch.Tensor, dtype=float32, shape=(*L, C)): sampled actions
            act_inds (torch.Tensor, dtype=int64, shape=(*L, C)): sampled action indices
        """
        act_inds = super().sample()  # (*L, C) dtype=torch.int64
        act_inds_unsqueezed = act_inds.unsqueeze(-1)  # (*L, C, 1)

        act = self.actions.gather(dim=-1, index=act_inds_unsqueezed)  # (*L, C, 1)
        act = act.squeeze(-1)  # (*L, C)
        return act, act_inds

    def argmax(self):
        """
        Returns:
            act (torch.Tensor, dtype=float32, shape=(*L, C)): most likely actions
            act_inds (torch.Tensor, dtype=int64, shape=(*L, C)): most likely action indices
        """
        act_inds = self.logits.argmax(dim=-1)  # (*L, C)  dtype=torch.int64
        act_inds_unsqueezed = act_inds.unsqueeze(-1)  # (*L, C, 1)

        act = self.actions.gather(dim=-1, index=act_inds_unsqueezed)  # (*L, C, 1)
        act = act.squeeze(-1)  # (*L, C)
        return act, act_inds

    def log_prob(self, action_inds):
        """
        Args:
            action_inds (torch.Tensor, dtype=int64, shape=(*L, C)): batch of action indices.

        Returns:
            logp (torch.Tensor, dtype=float32, shape=(*L)): log probabilities, treating action distribution as
                independent across camera rays. This effectively sums the log probs across camera rays.
        """
        logp = super().log_prob(action_inds)  # (*L, C)
        logp = logp.sum(dim=-1)  # (*L)
        return logp

    def entropy(self):
        ent = super().entropy()  # (*L, C)
        ent = ent.mean(dim=-1)  # (*L)
        return ent
