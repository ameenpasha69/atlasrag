# Atlas Gateway Runbook

The Atlas Gateway is the ingress service that accepts carrier callbacks for Meridian
Logistics. Every gateway node is registered under the hardware asset tag MG-GW-7741 and
reports to the regional controller in Pune.

## Connection handling

The gateway accepts inbound webhooks over TLS 1.3 only. The default request timeout is 30
seconds, measured from the first byte of the request line to the last byte of the response
body. Operators may override the value per carrier profile, but the supported range is 5 to
120 seconds and values outside that range are rejected at configuration load.

Idle keep-alive connections are closed after 75 seconds. This is deliberately longer than
the request timeout so that a slow carrier does not lose its connection mid-retry.

## Retry semantics

A carrier callback that returns a 5xx status is retried with exponential backoff starting at
2 seconds and doubling up to a ceiling of 64 seconds. After the eighth attempt the callback
is written to the dead letter queue named gw-dlq-primary and an operator page is raised.

Callbacks that return 4xx are never retried, because a client error will not resolve itself.
The single exception is 429, which is retried using the interval supplied in the Retry-After
header when that header is present and parseable.

## Failure modes seen in production

When the controller in Pune is unreachable the gateway continues to accept callbacks and
buffers them locally for up to 15 minutes. Beyond that window the buffer is flushed to disk
and the node reports itself unhealthy on its readiness endpoint, which removes it from the
load balancer pool without terminating in-flight requests.

A full local buffer is the most common cause of elevated error rates during controller
maintenance. The mitigation is to drain nodes one at a time rather than restarting the
controller while every node holds a buffer.

## Health and readiness

The liveness endpoint reports only whether the process is running. The readiness endpoint
additionally checks controller reachability, disk headroom on the buffer volume, and the
age of the oldest buffered callback. Confusing the two is the most common operator error
during an incident, because a live process with an unreachable controller still reports
healthy on liveness.
