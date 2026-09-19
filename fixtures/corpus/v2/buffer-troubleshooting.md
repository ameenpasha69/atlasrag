# Buffer and Dead Letter Troubleshooting

This guide covers failures of the local callback buffer and of the dead letter queue. Handshake
and carrier certificate failures are covered in the gateway error code reference instead.

## ERR-4471 — Buffer volume exhausted

The local callback buffer volume has no remaining headroom. The node reports itself unhealthy
on the readiness endpoint and is removed from the load balancer pool, so no new callbacks are
routed to it while the condition persists.

Action: drain the node and extend the buffer volume. Restarting the agent without adding
capacity will simply reproduce the condition within minutes, and each restart loses whatever
was still held in memory.

## ERR-7441 — Dead letter queue write failed

A callback exhausted its retries and the subsequent write to gw-dlq-primary also failed. This
is the only condition in either reference that can result in a permanently lost callback.

Action: page the on-call operator immediately. Check queue capacity before replaying anything,
because a replay into a full queue reproduces the same failure and doubles the volume to
investigate afterwards.

## Prevention

Buffer headroom is checked hourly and alerts at seventy percent. The alert exists because the
gap between seventy and one hundred percent has been as short as forty minutes during a
controller outage.
