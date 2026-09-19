# Incident Report 2026-03-14: Dispatch Log Gap

## Summary

Between 2026-03-11 and 2026-03-14 the archival job for dispatch logs failed silently. The
failure was discovered when an auditor requested records from the previous quarter and the
export returned fewer rows than expected.

## Retention as operated

Dispatch logs are retained for 90 days before archival. The archival job copies them to cold
storage and then removes them from the hot table. Because the job failed, logs older than the
retention window were neither archived nor removed, and the hot table grew by approximately
14 gigabytes over the incident window.

## Root cause

The archival job authenticates with a service credential that expired on 2026-03-11. The job
treated the authentication failure as a transient error, logged it at debug level, and exited
with status zero. Nothing in the alerting pipeline inspects debug level output, so no alert
fired.

## Corrective actions

The job now exits non-zero on any authentication failure, and the alerting pipeline treats a
missing success heartbeat as a page-worthy condition. A separate ticket tracks rotating the
service credential automatically rather than manually.
