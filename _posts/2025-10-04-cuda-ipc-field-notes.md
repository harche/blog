---
layout: post
title: "CUDA IPC Field Notes from the Trenches"
description: "Every experiment that finally made GPU memory sharing work across processes and pods"
date: 2025-10-04
author: Harshal Patil
---

*Author: [Harshal Patil](https://github.com/harche){:target="_blank" rel="noopener"}*

I opened the repo [`harche/cuda-ipc-debugging`](https://github.com/harche/cuda-ipc-debugging){:target="_blank" rel="noopener"} to keep track of the ways CUDA inter-process communication (IPC) behaves when you throw it at bare metal, at two pods shouting across an `emptyDir`, or at Kubernetes Dynamic Resource Allocation (DRA).

This post is long on purpose. Each section explains **what we tried, how to run it yourself, the exact security knobs that matter,** and the expected output (including the failure modes we tripped over). If you’re debugging CUDA IPC on a cluster, consider this your field guide.

---

## Lab Setup and Reusable Bits

- **GPU hardware**: Any NVIDIA GPU that supports peer access. I ran most tests on L4s but the patterns hold for A100s as well.
- **Driver/runtime**: NVIDIA container runtime (`nvidia-container-toolkit`) with CUDA 12.x.
- **Cluster baseline**: Kubernetes 1.31 for the DRA scenarios; earlier versions work for the non-DRA experiments.

If you want to check topology, compile and run `ipc_example/peer_access_matrix.cu`. It prints a matrix of `cudaDeviceCanAccessPeer(i, j)` values so you can confirm which GPU pairs support peer access.

---

## Phase 1 — Bare-Metal Reality Check (`ipc_example/`)

Before Kubernetes enters the chat, we run the classic two-process dance on a single node. The directory contains:

- `example.cu`: Producer + consumer logic using POSIX shared memory to exchange a `cudaIpcMemHandle_t`.
- `peer_access_matrix.cu`: Optional diagnostic for GPU peer access.
- `README.md`: Minimal compile/runtime instructions.

### Run Book

```bash
cd ipc_example
nvcc example.cu -o example

# Terminal A (producer)
./example

# Terminal B (consumer)
./example 0
```

What happens under the hood:

1. **Producer** allocates 1 MB on the GPU, zeroes it with `cudaMemset`, and writes the IPC handle into a Linux shared-memory segment (`shm_open`).
2. **Consumer** waits for the shared-memory flag, then `cudaIpcOpenMemHandle` maps the same GPU memory into its address space and fills it with `1`s.
3. Producer wakes up, copies the buffer back to host, and should read a sum of `DATA_SIZE`.

If the consumer opens the handle before the producer has flagged `handle_ready`, you’ll see an infinite wait. That’s expected—it’s a reminder that **ordering matters** for every scenario that follows.

### Common Gotchas

- `cudaIpcOpenMemHandle: invalid argument` → you built against a different CUDA version than the driver supports.
- Sum of zeros → consumer never ran, or couldn’t open the handle. Check the shared-memory file (`/dev/shm/linux_shm`).

This simple program is the canary. If this fails, don’t even bother with Kubernetes yet.

---

## Phase 2 — Two Pods and a Shared Volume (`shared-volume-example/`)

Goal: prove two separate pods on the same node can share GPU memory via CUDA IPC. We wire them together with an `emptyDir` volume that carries both the IPC handle and a "ready" flag.

### Files

- `producer-pod.yaml`
- `consumer-pod.yaml`
- `README.md` (contains full log snippets and a security matrix)

### Deployment

```bash
kubectl apply -f shared-volume-example/producer-pod.yaml
kubectl wait --for=condition=Ready pod/cuda-ipc-producer-simple --timeout=60s
kubectl apply -f shared-volume-example/consumer-pod.yaml
kubectl wait --for=condition=Ready pod/cuda-ipc-consumer-simple --timeout=60s
```

**Important sequencing:** producer first, then consumer. The consumer reads the IPC handle from `/mnt/ipc/handle`. Start it early and you’ll hit `ENOENT`.

### Security Knob Truth Table

| HostIPC | HostPID | Privileged | Result | Error |
| --- | --- | --- | --- | --- |
| ✅ | ✅ | ✅ | ✅ success | — |
| ❌ | ✅ | ✅ | ✅ success | — |
| ✅ | ❌ | ✅ | ❌ fail | `invalid device context` |
| ✅ | ✅ | ❌ | ❌ fail | `invalid argument` |
| ❌ | ❌ | ✅ | ❌ fail | `invalid device context` |
| ❌ | ✅ | ❌ | ❌ fail | `invalid argument` |
| ✅ | ❌ | ❌ | ❌ fail | `invalid device context` |
| ❌ | ❌ | ❌ | ❌ fail | `invalid device context` |

**Key takeaways**:

- You can disable **HostIPC** outright. The shared `emptyDir` is enough for signalling.
- You **must** give the containers **HostPID or shareProcessNamespace** so the CUDA driver can attach to the same GPU context.
- You **must** run them **privileged** or the device nodes will be restricted and `cudaIpcOpenMemHandle` bails out.

### Happy Path Logs

Producer (`cuda-ipc-producer-simple`):

```
Producer: Allocating GPU memory...
Producer: Creating IPC handle...
Producer: Writing handle to shared volume...
Producer: Ready signal sent. Hanging infinitely to keep memory alive...
```

Consumer (`cuda-ipc-consumer-simple`):

```
Consumer: Opening IPC memory handle...
Consumer: Successfully opened shared GPU memory!
Consumer: First 10 values from shared memory: 42 43 44 45 46 47 48 49 50 51
Consumer: ✓ Data verification PASSED!
```

Seeing anything else? Check the security settings first; nine times out of ten it’s that.

To tear down:

```bash
kubectl delete pod cuda-ipc-producer-simple
kubectl delete pod cuda-ipc-consumer-simple
```

---

## Phase 3 — Single Pod, Two Containers (`single-pod-example/`)

This scenario runs producer and consumer side-by-side in a single pod. Advantages: fewer manifests, shared lifecycle, and easier log tailing. The core mechanics are the same (`emptyDir` + handshake files), but the security posture changes.

### Manifest Highlights

```yaml
spec:
  shareProcessNamespace: true   # safer alternative to hostPID
  containers:
    - name: producer
      securityContext:
        privileged: true
    - name: consumer
      securityContext:
        privileged: true
```

We leave `hostIPC` off, and instead of toggling host PID access we flip `shareProcessNamespace: true`. That keeps processes visible to both containers without joining the host PID namespace.

### Deploy It

```bash
kubectl apply -f single-pod-example/cuda-ipc-pod-concurrent.yaml
kubectl wait --for=condition=Ready pod/cuda-ipc-concurrent-pod --timeout=60s
```

Inspect logs:

```bash
kubectl logs cuda-ipc-concurrent-pod -c producer | head
kubectl logs cuda-ipc-concurrent-pod -c consumer | head
```

Expected consumer output mirrors the two-pod case (`✓ Data verification PASSED!`). If you remove `shareProcessNamespace` *and* `hostPID`, you revert to `invalid device context` errors. Privileged mode is still required—the pod needs full `/dev/nvidia*` access.

### Why shareProcessNamespace Works

CUDA IPC relies on process-level handles. `shareProcessNamespace: true` lets both containers discover each other’s PIDs and GPU contexts without exposing the host’s PID namespace. It’s the best balance of functionality and isolation we found for single-pod deployments.

Cleanup:

```bash
kubectl delete pod cuda-ipc-concurrent-pod
```

---

## Phase 4 — Dynamic Resource Allocation (`single-pod-dra-example/`)

Kubernetes 1.31 introduced Dynamic Resource Allocation (DRA). With NVIDIA’s DRA driver (`dra-shared-gpu-example/` in the repo has the setup manifests) you can request virtualised GPU slices and, crucially, **drop privileged mode**.

### Key Files

- `resource-claim.yaml`: Creates a `ResourceClaim` for two shared GPUs.
- `single-pod-dra.yaml`: Pod manifest requesting the claim.
- `README.md`: Full security analysis and expected logs.

### Deployment Steps

```bash
kubectl apply -f dra-shared-gpu-example/nvidia-driver.yaml    # if you haven’t already
kubectl apply -f single-pod-dra-example/resource-claim.yaml
kubectl apply -f single-pod-dra-example/single-pod-dra.yaml
kubectl wait --for=condition=Ready pod/cuda-ipc-single-pod-dra -n cuda-ipc-single-dra --timeout=60s
```

### Security Matrix (DRA)

| HostIPC | HostPID | shareProcessNamespace | Privileged | Result |
| --- | --- | --- | --- | --- |
| ✅ | ✅ | ❌ | ✅ | ✅ success (baseline) |
| ❌ | ✅ | ❌ | ✅ | ✅ success |
| ✅ | ❌ | ✅ | ✅ | ✅ success |
| ✅ | ✅ | ❌ | ❌ | ✅ success (thanks to DRA) |
| ✅ | ❌ | ✅ | ❌ | ✅ success |
| ❌ | ❌ | ✅ | ❌ | ✅ success |
| ❌ | ❌ | ❌ | ❌ | ❌ fail (`invalid device context`) |

**Headline:** DRA lets us remove privileged mode entirely, provided we still give the containers shared process visibility (`hostPID` or `shareProcessNamespace`). The DRA driver handles device brokering and exposes the necessary file descriptors in a controlled fashion.

### GPU Selection Strategy

Both containers set `CUDA_VISIBLE_DEVICES="0,1"`. In code we explicitly pick devices:

```cuda
// Producer
cudaSetDevice(0);
cudaIpcGetMemHandle(&handle, devPtr);

// Consumer
cudaSetDevice(1);
cudaIpcOpenMemHandle(&devPtr, handle, cudaIpcMemLazyEnablePeerAccess);
```

The logs confirm it:

```
Producer: Found 2 GPU(s)
Producer: GPU 0: NVIDIA L4
Producer: GPU 1: NVIDIA L4
Consumer: Found 2 GPU(s)
Consumer: GPU 0: NVIDIA L4
Consumer: GPU 1: NVIDIA L4
Consumer: ✓ Data verification PASSED!
```

### Monitoring Tips

```bash
kubectl get pods -n cuda-ipc-single-dra
kubectl get resourceclaim -n cuda-ipc-single-dra
kubectl describe resourceclaim single-pod-dual-gpus -n cuda-ipc-single-dra
kubectl logs cuda-ipc-single-pod-dra -n cuda-ipc-single-dra -c producer | head
kubectl logs cuda-ipc-single-pod-dra -n cuda-ipc-single-dra -c consumer | head
```

Cleanup:

```bash
kubectl delete -f single-pod-dra-example/single-pod-dra.yaml
kubectl delete -f single-pod-dra-example/resource-claim.yaml
```

---

## Side Quest — DRA Shared GPU Driver (`dra-shared-gpu-example/`)

If you’re new to DRA, the `dra-shared-gpu-example/` directory contains:

- A Helm chart snippet for the NVIDIA DRA driver.
- Sample workloads showing two pods sharing one physical GPU safely.
- Notes on matching driver versions between host, container toolkit, and the DRA plugin.

Set this up first; nothing in the DRA section works without it.

---

## Troubleshooting Playbook

| Symptom | Likely Cause | Fix |
| --- | --- | --- |
| `cudaIpcOpenMemHandle: invalid device context` | No shared PID namespace | Enable `hostPID: true` or `shareProcessNamespace: true` |
| `cudaIpcOpenMemHandle: invalid argument` | Not privileged (non-DRA paths) | Add `securityContext.privileged: true` |
| Consumer hangs waiting for handle | Producer never wrote the ready flag | Check producer logs; ensure it’s running first |
| Pod stays `Init` in DRA scenario | ResourceClaim not bound | `kubectl describe resourceclaim ...` to debug driver deployment |
| Sum is zero in bare-metal example | Consumer never executed or failed silently | Check consumer stdout/stderr |

If all else fails, run `nvidia-smi` inside each container to confirm the device list. It saves a lot of time when the scheduler quietly lands pods on different nodes.

---

## Future Experiments

1. **Smarter handshakes**: Replace the file-based ready signal with gRPC or a lightweight REST callback so the producer can exit when the consumer finishes.
2. **Better metrics**: Add Prometheus exporters for IPC latency and throughput to quantify the impact of DRA vs. traditional setups.
3. **Cross-node IPC**: Investigate NVSwitch/NVLink + GPUDirect RDMA to see how far IPC handles can travel.
4. **Device awareness**: Extend `peer_access_matrix.cu` to print `cudaDeviceProp` so you know exactly which SKU each index maps to.

Got improvements or horror stories? Open an issue or PR on the repo. CUDA IPC still has sharp edges, but with these experiments you at least know which gloves to wear.
