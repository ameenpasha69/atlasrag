# Courier Dispatch Overview

Meridian operates three regional hubs. Each hub receives inbound freight through an intake
door, moves it across a staging belt, and feeds it into a sorting line that assigns every
item to an outbound courier route.

## Where work accumulates

Parcels accumulate at the staging belt whenever the sorting line processes fewer items per
minute than the intake door admits. The belt has no overflow lane, so the accumulation is
visible to the floor supervisor within a few minutes and is the earliest warning that the
hub is falling behind schedule.

Supervisors resolve the imbalance either by slowing the intake door or by opening a second
sorting line. Slowing intake is preferred during the evening surge because opening a second
line requires a crew reassignment that takes roughly eleven minutes to complete.

## Route assignment

Route assignment is deterministic. Each item carries a destination postal code, and the
assignment table maps every postal code to exactly one outbound courier route per shift.
Two items with the same postal code in the same shift therefore always travel together.

When a postal code has no mapping, the item is diverted to the exception lane and handled
manually. Unmapped postal codes are the second most common source of exception lane volume
after damaged labels.

## Shift structure

Hubs run two shifts. The morning shift begins at 06:00 and the evening shift at 14:00, with
a thirty minute handover during which both crews are present. Throughput during handover is
intentionally reduced so that the outgoing crew can complete open exception lane items
rather than passing partially handled freight to the incoming crew.
