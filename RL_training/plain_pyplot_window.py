import time
from typing import List, Optional

import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback


class PlottingPPO(PPO):
    """Standard PPO with exposed latest training loss for a normal pyplot window."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.latest_train_loss: Optional[float] = None
        self.loss_history: List[float] = []

    def train(self) -> None:
        super().train()
        loss = None
        try:
            values = getattr(self.logger, "name_to_value", {})
            if isinstance(values, dict):
                loss = values.get("train/loss", None)
                if loss is None:
                    loss = values.get("train/value_loss", None)
        except Exception:
            loss = None

        if loss is not None:
            try:
                loss = float(loss)
            except Exception:
                loss = None

        self.latest_train_loss = loss
        if loss is not None:
            self.loss_history.append(loss)


class LivePyplotCallback(BaseCallback):
    """Regular live matplotlib window: training loss + episode MATCH %."""

    def __init__(self, refresh_every_steps: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.refresh_every_steps = max(1, int(refresh_every_steps))
        self.loss_x: List[int] = []
        self.loss_y: List[float] = []
        self.match_x: List[int] = []
        self.match_y: List[float] = []
        self._last_loss_seen: Optional[float] = None
        self._window_ready = False
        self._last_draw_t = 0.0

    def _setup_window(self):
        plt.ion()
        self.fig, (self.ax_loss, self.ax_match) = plt.subplots(2, 1, figsize=(10, 7))
        try:
            self.fig.canvas.manager.set_window_title("RL Live Training")
        except Exception:
            pass

        self.loss_line, = self.ax_loss.plot([], [], linewidth=2)
        self.match_line, = self.ax_match.plot([], [], linewidth=2)

        self.ax_loss.set_title("Training Loss")
        self.ax_loss.set_xlabel("Timesteps")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.grid(True)

        self.ax_match.set_title("MATCH %")
        self.ax_match.set_xlabel("Timesteps")
        self.ax_match.set_ylabel("Percent")
        self.ax_match.set_ylim(0, 100)
        self.ax_match.grid(True)

        self.fig.tight_layout()
        self._window_ready = True
        plt.show(block=False)
        plt.pause(0.001)

    def _append_loss_if_new(self):
        loss = getattr(self.model, "latest_train_loss", None)
        if loss is None:
            return
        if self._last_loss_seen is not None and float(loss) == float(self._last_loss_seen):
            return
        self._last_loss_seen = float(loss)
        self.loss_x.append(int(self.model.num_timesteps))
        self.loss_y.append(float(loss))

    def _append_match_from_infos(self):
        infos = self.locals.get("infos", [])
        if not infos:
            return
        for info in infos:
            if not isinstance(info, dict):
                continue
            if not info.get("episode_done", False):
                continue
            match_pct = info.get("match_pct", None)
            if match_pct is None:
                continue
            self.match_x.append(int(self.model.num_timesteps))
            self.match_y.append(float(match_pct))

    def _redraw(self, force: bool = False):
        if not self._window_ready:
            self._setup_window()

        now = time.time()
        if not force and (now - self._last_draw_t) < 0.08:
            return
        self._last_draw_t = now

        if self.loss_x:
            self.loss_line.set_data(self.loss_x, self.loss_y)
            self.ax_loss.relim()
            self.ax_loss.autoscale_view()

        if self.match_x:
            self.match_line.set_data(self.match_x, self.match_y)
            self.ax_match.relim()
            self.ax_match.autoscale_view(scalex=True, scaley=False)
            self.ax_match.set_ylim(0, 100)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def _on_training_start(self) -> None:
        self._setup_window()

    def _on_step(self) -> bool:
        self._append_match_from_infos()
        if self.n_calls % self.refresh_every_steps == 0:
            self._append_loss_if_new()
            self._redraw()
        return True

    def _on_rollout_end(self) -> None:
        self._append_loss_if_new()
        self._redraw(force=True)

    def _on_training_end(self) -> None:
        self._append_loss_if_new()
        self._redraw(force=True)
        plt.ioff()
        plt.show(block=False)
