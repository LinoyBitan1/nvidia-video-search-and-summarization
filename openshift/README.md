# Deploying NVIDIA VSS on Red Hat OpenShift AI

This guide covers deploying the [NVIDIA Video Search and Summarization (VSS) v2.4.1](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) blueprint on Red Hat OpenShift AI (RHOAI) using a single Helm command. All OpenShift-specific adaptations are applied at install time - no post-deploy patching is required.

## Table of Contents

- [What We're Deploying](#what-were-deploying)
- [Tested Hardware](#tested-hardware)
- [Prerequisites](#prerequisites)
- [Configuration Reference](#configuration-reference)
- [Deployment](#deployment)
- [Verification](#verification)
- [Accessing the UI](#accessing-the-ui)
- [Model Size Optimization](#model-size-optimization)
- [OpenShift-Specific Challenges and Solutions](#openshift-specific-challenges-and-solutions)
- [Deployment Files](#deployment-files)

---

## What We're Deploying

VSS is a video analytics platform that ingests video (file upload or RTSP live stream), captions individual frames using a Vision Language Model, and makes the content searchable via natural language. It combines:

- **Vision Language Model** (Cosmos-Reason2-8B) for frame-by-frame video captioning
- **RAG pipeline** with vector search (Milvus), embedding, and reranking for natural language queries
- **LLM inference** for chat, summaries, and notifications
- **Event alerting** triggered when captions match user-defined keywords
- **Graph and document databases** (ArangoDB, Neo4j, Elasticsearch) for metadata and knowledge

**Data flow:**

- **Upload:** video → vss (Cosmos captions each frame) → nemo-embedding → Milvus
- **Search:** user query → nemo-embedding → Milvus (vector search) → nemo-rerank → nim-llm → response
- **Alerts:** vss monitors captions for user-defined event keywords

The Helm chart deploys 11 pods. Four require GPUs (`vss`, `nim-llm`, `nemo-embedding`, `nemo-rerank`); the rest are infrastructure services (Milvus, MinIO, etcd, Elasticsearch, ArangoDB, Neo4j).

---

## Tested Hardware

This deployment was validated on the following cluster configuration:

**Cluster:** OpenShift 4.19 on AWS (us-east-2)

### GPU nodes

| Instance Type | GPU | VRAM | vCPU | RAM | Count | Role in VSS |
|---------------|-----|------|------|-----|-------|-------------|
| `g6e.2xlarge` | 1x NVIDIA L40S | 46 GB | 8 | 64 GiB | 2 | VLM (Cosmos-Reason2-8B), nemo-rerank (1 GPU each) |
| `p4d.24xlarge` | 8x NVIDIA A100 40GB | 40 GB each | 96 | 1.1 TiB | 1 | nim-llm (Llama 8B), nemo-embedding (2 of 8 GPUs used) |

### Worker nodes (non-GPU)

| Instance Type | vCPU | RAM | Count | Role in VSS |
|---------------|------|-----|-------|-------------|
| `m6i.2xlarge` | 8 | 32 GiB | 5 | Milvus, MinIO, etcd, Elasticsearch, ArangoDB, Neo4j |

### Minimum hardware for reproduction

Any cluster with the following should work:

- **4 GPUs** with at least **40 GB VRAM** each (L40S, A100, or equivalent) — one each for VLM, nim-llm, nemo-embedding, nemo-rerank. NVIDIA A10G (22 GB) is **not sufficient** — the VLM (Cosmos-Reason2-8B) requires ~22 GiB for model weights + KV cache, exceeding available memory
- **~1 CPU core** and **~17 GiB RAM** across worker nodes for non-GPU pods (Elasticsearch alone requests 16 GiB)
- To run **Llama 70B** instead of 8B, nim-llm requires **4 GPUs on a single node** (tensor parallelism cannot span nodes), plus 2 GPUs for the VLM (upstream default), for a total of **8 GPUs**. NVIDIA recommends A100 **80GB** or higher for 70B. Set `LLM_MODEL=meta/llama-3.1-70b-instruct`, `LLM_IMAGE=nvcr.io/nim/meta/llama-3.1-70b-instruct`, `LLM_GPU_COUNT=4`, `VLM_GPU_COUNT=2`. See NVIDIA's [supported platforms](https://docs.nvidia.com/vss/latest/content/supported_platforms.html#supported-platforms) for validated GPU topologies

---

## Prerequisites

- OpenShift CLI (`oc`) 4.12+ installed and authenticated with cluster-admin privileges
- Helm 3.x installed
- NVIDIA GPU Operator installed on the cluster and `nvidia.com/gpu` resource is allocatable
- NGC API key from [NGC](https://org.ngc.nvidia.com/setup/api-keys) or [build.nvidia.com](https://build.nvidia.com/) (requires NVIDIA AI Enterprise license)
- HuggingFace token with the [Cosmos-Reason2-8B license](https://huggingface.co/nvidia/Cosmos-Reason2-8B) accepted
- GPU nodes are ready: `oc get nodes -l nvidia.com/gpu`
- GPU node taint keys identified: `oc describe node <gpu-node> | grep -A5 Taints`
- Helm chart `deploy/helm/nvidia-blueprint-vss-2.4.1.tgz` present in this repo

---

## Configuration Reference

All options are set via environment variables before calling the deploy script.

### Required Variables


| Variable      | Description                                             |
| ------------- | ------------------------------------------------------- |
| `NAMESPACE`   | Kubernetes namespace to deploy into                     |
| `NGC_API_KEY` | NGC API key for image pulls and NIM authentication      |
| `HF_TOKEN`    | HuggingFace token for the gated Cosmos-Reason2-8B model |


### Optional Variables


| Variable                | Default                                  | Description                                                                                  |
| ----------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------- |
| `LLM_MODEL`             | `meta/llama-3.1-8b-instruct`             | NIM LLM model name                                                                           |
| `LLM_IMAGE`             | `nvcr.io/nim/meta/llama-3.1-8b-instruct` | NIM LLM container image (must correspond to `LLM_MODEL`)                                     |
| `LLM_IMAGE_TAG`         | `latest`                                 | NIM LLM image tag                                                                            |
| `LLM_GPU_COUNT`         | `1`                                      | GPUs allocated to nim-llm                                                                    |
| `VLM_GPU_COUNT`         | `1`                                      | GPUs allocated to the vss VLM (Cosmos-Reason2-8B)                                            |
| `GPU_TOLERATION_KEYS`   | `nvidia.com/gpu`                         | Comma-separated GPU node taint keys                                                          |
| `GPU_TOLERATION_EFFECT` | `NoSchedule`                             | Toleration effect matching the GPU node taint                                                |
| `DISABLE_GUARDRAILS`    | `false`                                  | Set `true` to disable input guardrails (workaround for false positives on multi-image input) |


---

## Deployment

```bash
NGC_API_KEY=nvapi-... \
HF_TOKEN=hf_... \
NAMESPACE=<your-namespace> \
GPU_TOLERATION_KEYS=<taint-key> \
bash openshift/deploy-openshift.sh
```

Replace `GPU_TOLERATION_KEYS` with the actual taint key(s) on your GPU nodes (comma-separated for multiple taints). To find them:

```bash
oc describe node <gpu-node> | grep -A5 Taints
```

The script will:

1. Create the namespace if it does not exist
2. Create the `vss-sa` service account and grant the `anyuid` SCC
3. Pre-create all required secrets (`ngc-docker-reg-secret`, `ngc-api-key-secret`, `arango-db-creds-secret`, `minio-creds-secret`, `graph-db-creds-secret`, `hf-token-secret`)
4. Run `helm upgrade --install` with all OpenShift overrides
5. Wait for all Deployments and StatefulSets to be ready (up to 30 min - GPU pods download model weights on first run)
6. Expose the VSS UI via an OpenShift Route

---

## Verification

After the script exits, confirm all pods are running:

```bash
oc get pods -n <your-namespace>
```

All pods should be `Running` with `READY 1/1`. GPU pods (`nim-llm`, `nemo-embedding`, `nemo-rerank`, `vss`) may take 20-30 minutes on first deploy while model weights are downloaded and cached.

To follow progress on a specific pod:

```bash
oc logs -f deployment/vss-vss-deployment -n <your-namespace>
oc logs -f statefulset/nim-llm -n <your-namespace>
```

---

## Accessing the UI

The deploy script prints the UI URL at the end of the run:

```
=== Done ===
UI: http://<route-host>
```

Open the printed URL in a browser.

---

## Model Size Optimization

In GPU-constrained environments, the upstream chart's 70B LLM (4 GPUs) and 2-GPU VLM defaults leave multiple pods `Pending`. The deploy script overrides these to `llama-3.1-8b-instruct` (1 GPU) and 1 GPU for the VLM respectively.

The deploy script exposes configurable variables that propagate consistently at install time:

1. **LLM model and GPU count** - `LLM_MODEL`, `LLM_IMAGE`, and `LLM_GPU_COUNT` switch the image, resource request, and model name in a single operation. Defaults to `meta/llama-3.1-8b-instruct` with 1 GPU.
2. **VLM GPU count** - `VLM_GPU_COUNT` overrides the default 2-GPU request. The quantized `int4_awq` model fits on a single GPU.

Changing the LLM model also requires updating the model name in multiple locations - see [Challenge 10](#10-llm-model-name-consistency) for details.

---

## OpenShift-Specific Challenges and Solutions

The upstream VSS Helm chart targets vanilla Kubernetes. Running it on OpenShift requires addressing incompatibilities across security contexts, storage permissions, secrets, GPU scheduling, and service configuration. All fixes are applied at install time by `deploy-openshift.sh` and `values-openshift.yaml` - no post-deploy patching is required.

---

### 1. Storage Permissions

OpenShift assigns a random UID (e.g. `1000660000`) to containers rather than the UID defined in the image. Because this UID does not own the container's data directories, both services fail on startup with permission errors.

**Affected Services:**

- **milvus-minio** - Object storage for Milvus (`/minio_data`)
- **milvus** - Vector database persistence (`/var/lib/milvus`)

**Solution:** Mount an `emptyDir` volume over each problematic path. OpenShift automatically sets GID 0 with group-write permissions on `emptyDir` volumes, making them writable by any assigned UID.

```yaml
milvus-minio:
  extraPodVolumes:
  - name: data-volume
    emptyDir: {}
  extraPodVolumeMounts:
  - name: data-volume
    mountPath: /minio_data

milvus:
  extraPodVolumes:
  - name: data-volume
    emptyDir: {}
  extraPodVolumeMounts:
  - name: data-volume
    mountPath: /var/lib/milvus
```

> **Note:** `emptyDir` data is lost on pod restart. Replace with `PersistentVolumeClaims` for production.

---

### 2. Security Context Constraints

OpenShift's default `restricted-v2` SCC requires containers to run as a UID within the namespace-assigned range. Several sub-charts hardcode specific UIDs that fall outside this range, causing pods to fail admission with `unable to validate against any security context constraint: provider "anyuid": Forbidden`.

**Affected Services:**

- **arango-db** - Graph database (image-defined UID)
- **neo4j** - Graph database (`runAsUser: 7474`)
- **vss** - Core pipeline service (`runAsUser: 1000`)

**Solution:** Create a dedicated `vss-sa` service account and grant the `anyuid` SCC exclusively to it, scoping the elevated permission to a single named identity rather than the namespace-wide `default` service account.

```bash
oc create serviceaccount vss-sa -n "$NAMESPACE"
oc adm policy add-scc-to-user anyuid -z vss-sa -n "$NAMESPACE"
```

```yaml
arango-db:
  serviceAccount:
    create: false
    name: vss-sa

neo4j:
  serviceAccount:
    create: false
    name: vss-sa

vss:
  serviceAccount:
    create: false
    name: vss-sa
```

---

### 3. Security Context Removal

The GPU containers (NIM and NeMo) are pre-configured with specific user/group IDs (`runAsUser: 1000`) that conflict with OpenShift's random UID allocation. Unlike the services in [Challenge 2](#2-security-context-constraints) that require their hardcoded UIDs, these containers work fine under any UID.

**GPU-Dependent Services:**

- **nim-llm** - `podSecurityContext.runAsUser: 1000`
- **nemo-embedding** - `securityContext.runAsUser: 1000`
- **nemo-rerank** - `securityContext.runAsUser: 1000`

**Solution:** Nullify the hardcoded security contexts in `values-openshift.yaml`, allowing OpenShift to assign its own UID via the `restricted-v2` SCC:

```yaml
nim-llm:
  podSecurityContext:
    runAsUser: null
    runAsGroup: null
    fsGroup: null

nemo-embedding:
  applicationSpecs:
    embedding-deployment:
      securityContext:
        runAsUser: null
        runAsGroup: null
        fsGroup: null

nemo-rerank:
  applicationSpecs:
    ranking-deployment:
      securityContext:
        runAsUser: null
        runAsGroup: null
```

---

### 4. GPU Scheduling

GPU nodes carry custom `NoSchedule` taints. Without matching tolerations, the scheduler cannot place GPU workloads on those nodes and the pods stay `Pending`.

**GPU-Dependent Services:**

- **nim-llm** - LLM inference
- **nemo-embedding** - Embedding model
- **nemo-rerank** - Reranking model
- **vss** - Core pipeline and VLM

**Solution:** The deploy script builds tolerations dynamically from the `GPU_TOLERATION_KEYS` environment variable and applies them to all four GPU services at install time.

---

### 5. Missing Secrets

The chart references multiple secrets that must exist prior to installation but provides no mechanism to create them. Without them, pods fail with `secret not found` on volume mounts or image pulls.

**Required Secrets:**

- **ngc-docker-reg-secret** - Image pull secret for `nvcr.io`
- **ngc-api-key-secret** - Runtime NGC authentication for nemo-embedding and nemo-rerank. This is separate from the pull secret because image pull secrets (`kubernetes.io/dockerconfigjson`) cannot be referenced as `secretKeyRef` env vars
- **arango-db-creds-secret** - ArangoDB credentials
- **minio-creds-secret** - MinIO access credentials
- **graph-db-creds-secret** - Neo4j credentials, mounted as files by the parent chart.
- **hf-token-secret** - HuggingFace token for gated model downloads (see [Challenge 7](#7-hf_token-for-gated-model))

**Solution:** The deploy script pre-creates all required secrets before `helm install`.

---

### 6. Shared Memory Limit

Both sub-charts run NVIDIA Triton Inference Server with a Python BLS backend, which relies on POSIX shared memory (`/dev/shm`) for IPC between the server process and Python stub processes. OpenShift's default 64 MB `/dev/shm` limit is insufficient under concurrent inference load, resulting in `Failed to initialize Python stub: No space left on device` and pod crashes under load (exit code 137).

**Affected Services:**

- **nemo-embedding** - Vector embedding generation
- **nemo-rerank** - Document reranking

**Solution:** Mount a `Memory`-backed `emptyDir` at `/dev/shm`:

```yaml
nemo-embedding:
  extraPodVolumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 2Gi
  extraPodVolumeMounts:
  - name: dshm
    mountPath: /dev/shm

nemo-rerank:
  extraPodVolumes:
  - name: dshm
    emptyDir:
      medium: Memory
      sizeLimit: 2Gi
  extraPodVolumeMounts:
  - name: dshm
    mountPath: /dev/shm
```

---

### 7. HF_TOKEN for Gated Model

The vss container downloads `nvidia/Cosmos-Reason2-8B` from HuggingFace at startup. This model is gated - users must accept NVIDIA's license and authenticate with an HF token. Without `HF_TOKEN`, the download fails silently and the server never opens port 8000, so the pod stays `Running` but the readiness probe never passes.

**Solution:** Create `hf-token-secret` from a valid HuggingFace token that has accepted the [Cosmos-Reason2-8B license](https://huggingface.co/nvidia/Cosmos-Reason2-8B):

```bash
oc create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
  -n "$NAMESPACE"
```

The chart already references `hf-token-secret` as an optional `secretKeyRef` - the secret being absent is what caused the silent failure.

---

### 8. Tokenizer Thread Pool Burst

Both services run Triton with a Python BLS backend. Triton spawns 16 stub processes simultaneously at startup, each invoking the HuggingFace fast tokenizer's `encode()` during initialization. The tokenizer is Rust-backed and uses the Rayon thread pool library, which initializes lazily and defaults to one thread per CPU. On a high-CPU node, this produces thousands of simultaneous `pthread_create()` calls. The Linux kernel returns `EAGAIN` to some of them, causing Rayon to panic rather than retry, and the pod enters a crash loop with 200+ restarts.

**Affected Services:**

- **nemo-embedding** - Embedding model serving
- **nemo-rerank** - Reranking model serving

**Solution:** Set `TOKENIZERS_PARALLELISM=false` on both containers to disable the tokenizer's internal parallelism:

```yaml
nemo-embedding:
  applicationSpecs:
    embedding-deployment:
      containers:
        embedding-container:
          env:
          - name: TOKENIZERS_PARALLELISM
            value: "false"

nemo-rerank:
  applicationSpecs:
    ranking-deployment:
      containers:
        ranking-container:
          env:
          - name: TOKENIZERS_PARALLELISM
            value: "false"
```

---

### 9. Guardrails False Positive on Image Input with 8B LLM

When using `llama-3.1-8b-instruct` as the guardrails LLM, image summarization requests are incorrectly blocked as unsafe.

**Workaround:** Set `DISABLE_GUARDRAILS=true` when running the deploy script. This does not affect core search, summarization, or alert functionality. It passes the following to Helm:

```bash
--set "global.ucfGlobalEnv[1].name=DISABLE_GUARDRAILS"
--set-string "global.ucfGlobalEnv[1].value=true"
```

---

### 10. LLM Model Name Consistency

The LLM model name is hardcoded in three independent locations within the chart. Switching the LLM (e.g. from 70B to 8B) without updating all three causes the vss context manager to return 404 errors and guardrails to fall back to NVIDIA's cloud API with 401 Unauthorized.

**Affected Locations:**

- `nim-llm.model.name` - the model identity used by the NIM server itself
- `LLM_MODEL` env var in vss - used by the context manager for chat, summarization, and notifications
- `guardrails_config.yaml` `models[0]` - used by NeMo Guardrails for its startup validation test

**Solution:** The deploy script propagates `LLM_MODEL` to all three locations at install time. The guardrails `models[0]` entry requires all four fields (`engine`, `model`, `type`, `parameters.base_url`) - omitting `engine` or `type` causes a pydantic `ValidationError`; omitting `base_url` causes guardrails to call NVIDIA's cloud API instead of the local nim-llm service (401 Unauthorized).

```bash
--set "nim-llm.model.name=$LLM_MODEL"
--set "global.ucfGlobalEnv[0].name=LLM_MODEL"
--set-string "global.ucfGlobalEnv[0].value=$LLM_MODEL"
--set "vss.configs.guardrails_config\.yaml.models[0].engine=nim"
--set-string "vss.configs.guardrails_config\.yaml.models[0].model=$LLM_MODEL"
--set "vss.configs.guardrails_config\.yaml.models[0].parameters.base_url=http://llm-nim-svc:8000/v1"
--set "vss.configs.guardrails_config\.yaml.models[0].type=main"
```

---

## Deployment Files

All OpenShift customizations are codified in two files in the `openshift/` folder. The upstream chart remains in `deploy/helm/` and is referenced in place:

- **`deploy-openshift.sh`** - Main deployment script. Creates the namespace, service account, secrets, and runs `helm upgrade --install` with all overrides.
- **`values-openshift.yaml`** - Helm values override for OpenShift. Contains structural overrides that are cleanest in YAML.
- **`nvidia-blueprint-vss-2.4.1.tgz`** - The packaged upstream Helm chart.

Dynamic values (GPU tolerations, model name, image, secrets) are passed via `--set` flags in the deploy script. Static structural overrides live in the values file.