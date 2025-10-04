---
layout: post
title: "Keeping Distributed Jobs in Lockstep"
description: "How LWS and Volcano enforce all-or-nothing scheduling"
date: 2025-10-03
author: Harshal Patil
---

*Author: [Harshal Patil](https://github.com/harche){:target="_blank" rel="noopener"}*

Distributed workloads love to act like a synchronized swim team: either everyone dives in together or someone belly flops alone. **Gang scheduling** is how you keep the choreography perfect. Using [LeaderWorkerSet (LWS)](https://github.com/kubernetes-sigs/lws){:target="_blank" rel="noopener"} together with the [Volcano scheduler](https://github.com/volcano-sh/volcano){:target="_blank" rel="noopener"}, we can see exactly how that works across a few real scenarios. The manifests and scripts in this companion sandbox — [`lws-gang-demo`](https://github.com/harche/lws-gang-demo){:target="_blank" rel="noopener"} — simply give us a convenient stage.

---

## Why Gang Scheduling Exists

Multi-pod jobs fall apart when only part of the team runs. Without coordination, Kubernetes may start one pod, leave three Pending, and your training run or inference pipeline stalls forever. Gang scheduling fixes that by treating the pods as one atomic unit. If every pod in the gang can be scheduled, they all launch together; if not, they all wait.

With LWS, we get leader/worker orchestration for stateful or distributed jobs. Pairing it with Volcano brings the “all-or-nothing” scheduling behaviour needed for reliability.


![Pods diving together to show synchronized scheduling]({{ '/assets/images/gang-scheduling/overview-20251004.png' | relative_url }})

---

## Setting the Stage

The sandbox uses a small [kind](https://github.com/kubernetes-sigs/kind){:target="_blank" rel="noopener"} cluster and a few setup steps:

1. **Create the environment** – `scripts/setup-cluster.sh` builds a four-node kind cluster, installs Volcano, installs LWS v0.7.0, and patches the LWS config so `gangSchedulingManagement.schedulerProvider` is set to `volcano`.
2. **Grant permissions** – `manifests/setup/volcano-rbac.yaml` gives the LWS controller the right to create and manage Volcano PodGroups.
3. **Apply workloads** – each scenario is defined in `manifests/examples/*.yaml`, and `scripts/run-demo.sh` or `scripts/verify-gang-scheduling.sh` guide you through them step by step.

That’s all scaffolding; the interesting part is watching what happens when pods try to schedule.

---

## Scenario 1 – All Pods Ready? Launch Together

![Visualization of all pods launching together under gang scheduling]({{ '/assets/images/gang-scheduling/scenario1-20251004.png' | relative_url }})

Manifest: `manifests/examples/gang-test.yaml`

- 1 leader + 3 workers, each tagged with `schedulerName: volcano`.
- LWS automatically creates a PodGroup with `minMember = 4`.
- Volcano keeps the entire set "Inqueue" until four slots open, then flips every pod to Running in one shot.

```bash
$ kubectl get podgroups -n gang-demo
NAME                      STATUS    MINMEMBER   RUNNINGS
gang-test-0-6c67cc9968    Running   4           4
```

Takeaway: the gang scheduler is invisible when capacity exists — everything still starts instantly, just with guardrails in place.

---

## Scenario 2 – Resource Pinch, Zero Partial Launches

![Illustration of pods waiting because resources are constrained]({{ '/assets/images/gang-scheduling/scenario2-20251004.png' | relative_url }})

Manifest: `manifests/examples/gang-constrained.yaml`

- Pods request 4 CPUs each; the script taints two worker nodes so only one remains usable.
- LWS still creates the PodGroup (`minMember = 4`, `minResources` equal to the summed requests).
- Volcano sees the gang cannot fit, so every pod stays Pending. Events show the PodGroup stuck with `Unschedulable` until more nodes free up.
- The moment a taint is removed, Volcano schedules all four pods atomically.

```bash
$ kubectl get pods -n gang-demo -l leaderworkerset.sigs.k8s.io/name=gang-constrained
NAME                   READY   STATUS    RESTARTS   AGE
gang-constrained-0     0/1     Pending   0          24s
gang-constrained-0-1   0/1     Pending   0          24s
gang-constrained-0-2   0/1     Pending   0          24s
gang-constrained-0-3   0/1     Pending   0          24s
```

```bash
$ kubectl get podgroups -n gang-demo
NAME                           STATUS    MINMEMBER   RUNNINGS
<pod-group-name>              Inqueue   4           <none>
```

```bash
$ kubectl get events --field-selector involvedObject.kind=PodGroup -n gang-demo
TYPE      REASON          MESSAGE
Warning   Unschedulable   4/4 tasks in gang unschedulable: pod group is not ready, 4 Pending, 4 minAvailable;
                             Pending: 1 Unschedulable, 3 Schedulable.
                             Origin reason is gang-constrained-0-3: 0/4 nodes are unavailable:
                             1 Insufficient cpu, 1 node(s) had untolerated taint {node-role.kubernetes.io/control-plane: },
                             2 node(s) had untolerated taint {test: blocked}.
```

When you free one of the tainted nodes, the entire gang launches together:

```bash
$ kubectl taint nodes llm-d-demo-worker2 test=blocked:NoSchedule-
$ kubectl get podgroups -n gang-demo
NAME                           STATUS    MINMEMBER   RUNNINGS
<pod-group-name>              Running   4           4
```

Takeaway: gang scheduling prevents the classic deadlock where one leader runs, waits forever for its workers, and hogs resources along the way.

---

## Scenario 3 – Default Scheduler and the Sad Trombone

![Pods out of sync to represent partial deployment]({{ '/assets/images/gang-scheduling/scenario3-20251004.png' | relative_url }})

Manifest: `manifests/examples/no-gang.yaml`

- Same resource profile, but this time it’s a plain Deployment using the default scheduler.
- One lucky pod finds room and starts Running; the rest sit Pending.
- If you compare this output to Scenario 2, the difference is obvious: no PodGroup, no atomicity, and a partially launched workload that can’t make progress.

```bash
$ kubectl get pods -n gang-demo -l app=no-gang
NAME                            READY   STATUS    RESTARTS   AGE
no-gang-test-6686c8476b-28n8p   1/1     Running   0          5s
no-gang-test-6686c8476b-4jmdw   0/1     Pending   0          5s
no-gang-test-6686c8476b-9vnwv   0/1     Pending   0          5s
no-gang-test-6686c8476b-fq92q   0/1     Pending   0          5s
```

Takeaway: gang scheduling isn’t a luxury — it’s the difference between a healthy rollout and a half-deployed mess.

---

## Try It Yourself

1. `./scripts/setup-cluster.sh`
2. `./scripts/run-demo.sh` (walks through the three scenarios interactively)
3. `./scripts/verify-gang-scheduling.sh` (grabs PodGroup fields and events for receipts)
4. Optional cleanup: `./scripts/cleanup.sh`

You can also poke through the live resources with `kubectl get podgroups -n gang-demo` or `kubectl describe podgroup <pod-group-name> -n gang-demo` to watch Volcano flip PodGroups from `Inqueue` to `Running`.

---

## What to Remember

- Gang scheduling is all about **all-or-nothing** deployments; anything less invites deadlocks.
- LWS v0.7.0+ adds the tooling you need to pair with a gang scheduler without heavy lifting.
- Volcano acts as the enforcement engine, respecting `minMember` and `minResources` before letting pods bind.
- The moment you remove gang scheduling, the safety net goes away and partial rollouts come back.

Keep that mental model handy next time someone wonders why their distributed workload got itself stuck in `Pending`. Gang scheduling, done right, keeps the whole crew moving together.
