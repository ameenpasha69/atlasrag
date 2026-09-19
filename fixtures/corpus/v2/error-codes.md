# Gateway Error Code Reference

Codes in this reference concern the controller handshake and inbound carrier connections.
Buffer and dead-letter failures are covered separately in the buffer troubleshooting guide.

Codes share a prefix and differ only in their digits, and transposed digits are a common source
of misdiagnosis, so each code is listed with a distinct meaning and a distinct action.

## ERR-4417 — Controller handshake rejected

The controller refused the node's registration handshake because the node clock is more than
ninety seconds from controller time. The node continues to accept callbacks and buffers them,
but it will not appear in the controller inventory.

Action: correct time synchronisation on the node, then restart the agent. Do not restart the
controller, which will not change the outcome and will disturb every other node.

## ERR-7414 — Carrier certificate chain incomplete

An inbound carrier presented a certificate without its intermediate chain. The connection is
refused before any request body is read, so no callback is lost.

Action: contact the carrier. No change on the gateway side will resolve an incomplete chain,
and disabling verification is not an approved workaround.

## Reporting

Every occurrence is counted per node per hour. A single occurrence of either code is not
actionable on its own; a sustained rate across more than one node indicates a controller or
carrier problem rather than a node problem.
