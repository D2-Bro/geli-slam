"""Drain tracker callbacks before destroying ROS entities or replacing the executor."""

from rclpy.executors import MultiThreadedExecutor


class TrackerExecutor(MultiThreadedExecutor):
    """Join submitted work when the spin-owning thread stops the executor.

    The installed ROS Humble executor waits for active callbacks in shutdown(),
    but does not join its worker pool. Already submitted handlers can still
    start afterwards. Drain that pool before the caller destroys a subscription.
    shutdown() must run between spin_once() calls, never from a worker callback.
    """

    def shutdown(self):
        stopped = super().shutdown()
        # Let queued handlers finish while every subscription is still alive.
        # Cancelling pool futures would leave their ROS entity execution flags set.
        self._executor.shutdown(wait=True)
        pending, self._futures = self._futures, []
        for task in pending:
            if task.done():
                task.result()
        return stopped
