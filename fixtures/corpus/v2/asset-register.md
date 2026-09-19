# Regional Asset Register

This register lists hardware assets across the three hubs. Asset tags deliberately share a
common prefix and differ by one or two characters, which is how they are printed on the
physical labels.

## Gateway nodes

| Asset tag | Location | Role |
|---|---|---|
| MG-GW-7741 | Pune | primary carrier callback ingress |
| MG-GW-7742 | Pune | standby carrier callback ingress, cold spare |
| MG-GW-7751 | Nagpur | primary carrier callback ingress |
| MG-GW-7752 | Nagpur | standby, currently decommissioned pending disposal |

MG-GW-7742 has never carried production traffic. It is powered but unconfigured, and the
network team keeps it out of the load balancer pool deliberately.

MG-GW-7751 serves the Nagpur hub and is the only gateway outside Pune with a direct controller
link rather than a relayed one.

## Metrology instruments

| Serial | Instrument | Status |
|---|---|---|
| CAL-X7-4421B | optical gap reference unit | in service, Pune metrology cabinet |
| CAL-X7-4422B | optical gap reference unit | withdrawn after a drop, awaiting recertification |
| CAL-X9-4421B | thermal probe reference | in service, Nagpur metrology cabinet |

CAL-X7-4422B was withdrawn from service after it was dropped during a quarterly calibration
round. It must not be used for conveyor sensor calibration until recertified.

CAL-X9-4421B measures temperature, not gap width. Using it for conveyor calibration is a
recorded procedural error and has happened twice.

## Disposal

Assets marked decommissioned are held for ninety days before disposal so that any pending
audit can inspect them. Disposal records live in the asset system, not in this register.
