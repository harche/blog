---
layout: post
title: "CUDA IPC Field Notes from the Trenches"
description: "A playful walk through shared-memory experiments, Kubernetes pods, and DRA adventures"
date: 2025-10-04
author: Harshal Patil
---

*Author: [Harshal Patil](https://github.com/harche){:target="_blank" rel="noopener"}*

I needed a place to stash every trick I learned while making CUDA’s inter-process communication behave inside (and outside) Kubernetes. So I started the repo [`harche/cuda-ipc-debugging`](https://github.com/harche/cuda-ipc-debugging){:target="_blank" rel="noopener"}. This post is the highlight reel—no corporate slides, just the experiments that actually worked, the ones that didn’t, and why.

> Quick note: the `kubernetes/` and `openshift/` folders in the repo are legacy dumps. Everything interesting lives in the other directories we’re about to tour.

---

## Warm-Up: CUDA IPC without the Cluster

Before throwing pods at the problem, I wanted a zero-distraction environment. The `ipc_example/` directory gives you exactly that:

```bash
cd ipc_example
nvcc example.cu -o example
# Terminal 1 (producer)
./example
# Terminal 2 (consumer)
./example 0
```

`example.cu` wires together two processes with POSIX shared memory and two CUDA calls you’ll end up memorising: `cudaIpcGetMemHandle` and `cudaIpcOpenMemHandle`. Process 1 allocates 1 MB on the GPU, sets it to zero, and waits. Process 2 opens the same memory, flips everything to `1`, and the first process later verifies the sum. Simple, noisy, and perfect for spotting basic IPC mistakes.

If you’re troubleshooting topology questions, `peer_access_matrix.cu` will bruteforce `cudaDeviceCanAccessPeer` across eight slots and tell you which GPU pairs can see each other. Handy when the hardware team swears NVLink is wired up and reality says otherwise.

---

## Two Pods, One Shared Volume

`shared-volume-example/` is where I proved CUDA IPC can work across *separate* pods, so long as they gossip via an `emptyDir` volume. The flow looks like this:

1. Producer pod launches, allocates GPU memory, writes test data, drops the IPC handle into `/mnt/ipc/handle`, and signals readiness.
2. Consumer pod starts next, reads the handle, opens the shared GPU memory, and verifies the numbers.

Security permutations matter here. After more retries than I’d like to admit, the winning combo was:

- `hostPID: true` **or** `shareProcessNamespace: true`
- `privileged: true`
- `hostIPC: false` was totally fine (surprise!)

Miss any of those and you’ll be greeted by `invalid device context` or `invalid argument` errors from `cudaIpcOpenMemHandle`. When it works, the consumer logs a triumphant `✓ Data verification PASSED!`, and the producer keeps hanging around to keep the memory alive.

---

## One Pod, Two Containers, Less Drama

Next experiment: ditch the second pod and run both roles inside a single pod. That’s `single-pod-example/`. The spec spins up two containers—`producer` and `consumer`—with a shared `emptyDir` and a simple handshake file.

Why bother? You get fewer moving pieces, and you also learn exactly which security knobs you can dial down:

- `hostPID: true` **or** `shareProcessNamespace: true` → *required*
- `privileged: true` → *non-negotiable in this setup*
- `hostIPC` → optional, disable it if you want

The neat trick was discovering `shareProcessNamespace: true` is a softer alternative to `hostPID: true`. You still get the process visibility CUDA IPC needs, without dumping your containers into the host’s PID namespace.

Deploying it feels familiar:

```bash
kubectl apply -f single-pod-example/cuda-ipc-pod-concurrent.yaml
kubectl wait --for=condition=Ready pod/cuda-ipc-concurrent-pod --timeout=60s
kubectl logs cuda-ipc-concurrent-pod -c consumer
```

When you see the consumer printing the first ten values (`42 43 44 ...`), you’re golden.

---

## Enter DRA: Sharing GPUs without Privileged Mode

Dynamic Resource Allocation (DRA) landed in Kubernetes 1.31, and the `single-pod-dra-example/` directory is me poking it with a stick. Same two-container pod, but now you request a shared GPU allocation through a `ResourceClaim`. The headline results:

- `privileged: false` finally works, because the NVIDIA DRA driver takes over device brokering.
- You still need `hostPID: true` **or** `shareProcessNamespace: true` to avoid the dreaded `invalid device context`.
- `CUDA_VISIBLE_DEVICES` is set to `"0,1"`, and the CUDA code explicitly picks GPU 0 for the producer and GPU 1 for the consumer. Cross-GPU IPC works, and the logs confirm both sides see two GPUs available.

Deployment checklist:

```bash
kubectl apply -f single-pod-dra-example/resource-claim.yaml
kubectl apply -f single-pod-dra-example/single-pod-dra.yaml
kubectl wait --for=condition=Ready pod/cuda-ipc-single-pod-dra -n cuda-ipc-single-dra --timeout=60s
```

Once it’s running, check the consumer logs—you’ll see the same `✓ Data verification PASSED!`, this time without needing privileged containers.

---

## Cheat Sheet

| Scenario | What works | What breaks |
| --- | --- | --- |
| Bare-metal (`ipc_example/`) | Two local processes, POSIX shared memory, NVCC | Forgetting to launch the consumer after the producer (it just hangs) |
| Shared pods (`shared-volume-example/`) | `hostPID` **or** `shareProcessNamespace`, plus `privileged` | Removing privileged mode → `invalid argument` |
| Single pod (`single-pod-example/`) | Same as above, but fewer manifests | Disabling both `hostPID` and `shareProcessNamespace` |
| DRA single pod (`single-pod-dra-example/`) | DRA + `shareProcessNamespace` + non-privileged containers | Forgetting the `ResourceClaim` or leaving `hostPID` + `shareProcessNamespace` both false |

---

## Where to Hack Next

- Add health probes so the producer can shut down once the consumer has verified the data.
- Swap the `emptyDir` handshake for gRPC or a tiny REST callback—no more “sleep and hope” loops.
- Extend `peer_access_matrix.cu` to read the actual device names (`cudaDeviceProp`).
- Try the same setups on different GPU SKUs (A100 vs L4) and compare peer access behaviour.

If you end up tweaking the repo or catch a corner case I missed, drop a PR or ping me. CUDA IPC has plenty of sharp edges, but at least now we’ve mapped where most of them are hiding.
