# Deploying NVIDIA VSS on Red Hat OpenShift AI

This guide covers deploying the [NVIDIA Video Search and Summarization (VSS) v2.4.1](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) blueprint on Red Hat OpenShift AI (RHOAI) using a single Helm command. All OpenShift-specific adaptations are applied at install time - no post-deploy patching is required.

## Table of Contents

- [What We're Deploying](#what-were-deploying)
- [OpenShift Overlay Strategy](#openshift-overlay-strategy)
- [Tested Hardware](#tested-hardware)
- [Prerequisites](#prerequisites)
- [Configuration Reference](#configuration-reference)
- [Deployment](#deployment)
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

## OpenShift Overlay Strategy

1. We introduced an `openshift.enabled` flag in the pre-existing [`values.yaml`](nvidia-blueprint-vss-2.4.1.tgz) file. This is the only non-additive change we made on top of the NVIDIA upstream codebase.
2. All the values and resources required for the OpenShift deployment are placed in two dedicated files: [`values-openshift.yaml`](values-openshift.yaml) and [`templates/openshift.yaml`](nvidia-blueprint-vss-2.4.1.tgz) (inside the chart).
3. Secrets, ServiceAccount, SCC RoleBinding, and Route are created by the Helm chart template (`templates/openshift.yaml`), gated by the `openshift.enabled` flag. API keys are passed via `--set` at install time; all other secrets use default values defined in `values-openshift.yaml`.
4. The deployment requires only two files and a single Helm command — no external scripts or separate OpenShift directory. See the [Deployment](#deployment) section below.

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
- To run **Llama 70B** instead of 8B, nim-llm requires **4 GPUs on a single node** (tensor parallelism cannot span nodes), plus 2 GPUs for the VLM (upstream default), for a total of **8 GPUs**. NVIDIA recommends A100 **80GB** or higher for 70B. Edit `values-openshift.yaml`: `nim-llm.model.name`, `nim-llm.image.repository`, `nim-llm.resources.limits.nvidia.com/gpu=4`, `vss.resources.limits.nvidia.com/gpu=2`, `global.ucfGlobalEnv[0].value`. See NVIDIA's [supported platforms](https://docs.nvidia.com/vss/latest/content/supported_platforms.html#supported-platforms) for validated GPU topologies

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

All configuration is defined in [`values-openshift.yaml`](values-openshift.yaml). Only API keys are passed via `--set` at install time.

### Required (passed via `--set`)

| Parameter | Description |
|-----------|-------------|
| `openshift.secrets.ngcApiKey` | NGC API key for image pulls and NIM authentication |
| `openshift.secrets.hfToken` | HuggingFace token for the gated Cosmos-Reason2-8B model |

### Configurable in `values-openshift.yaml`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `nim-llm.model.name` | `meta/llama-3.1-8b-instruct` | NIM LLM model name |
| `nim-llm.image.repository` | `nvcr.io/nim/meta/llama-3.1-8b-instruct` | NIM LLM container image |
| `nim-llm.image.tag` | `latest` | NIM LLM image tag |
| `nim-llm.resources.limits.nvidia.com/gpu` | `1` | GPUs allocated to nim-llm (4 for 70B) |
| `vss.resources.limits.nvidia.com/gpu` | `1` | GPUs allocated to the vss VLM (2 for 70B) |
| `global.ucfGlobalEnv[0].value` | `meta/llama-3.1-8b-instruct` | LLM_MODEL env var propagated to all pods |
| `global.ucfGlobalEnv[1].value` | `true` | DISABLE_GUARDRAILS (recommended `true` for 8B) |
| Tolerations (vss, nim-llm, nemo-*) | `nvidia.com/gpu` | GPU node taint keys — update for your cluster |

---

## Deployment

### 1. Prepare your environment

```bash
# Verify OpenShift connectivity
oc login --token=$OPENSHIFT_TOKEN --server=$OPENSHIFT_CLUSTER_URL
oc whoami
oc cluster-info

# Verify Helm is installed
helm version

# Create and switch to the deployment namespace
oc new-project vss
```

### 2. Configure secrets

Export your API keys:

```bash
export NGC_API_KEY="<your NGC key>"
export HF_TOKEN="<your HuggingFace token>"
```

**Option A: Let Helm create secrets (for development)**

The chart creates all secrets automatically. Set your API keys in `values-openshift.yaml` or pass them via `--set`:

```yaml
openshift:
  secrets:
    create: true
    ngcApiKey: "your-ngc-api-key"
    hfToken: "your-hf-token"
```

**Option B: Create secrets manually (recommended for production)**

> **Important:** Secret names are hardcoded in the chart's sub-charts. They must match exactly as shown below — regardless of the Helm release name.

```bash
# NGC container registry pull secret
oc create secret docker-registry ngc-docker-reg-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="$NGC_API_KEY" \
  -n vss

# NGC API key
oc create secret generic ngc-api-key-secret \
  --from-literal=NGC_API_KEY="$NGC_API_KEY" \
  -n vss

# HuggingFace token
oc create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="$HF_TOKEN" \
  -n vss

# ArangoDB credentials
oc create secret generic arango-db-creds-secret \
  --from-literal=username=root \
  --from-literal=password=password \
  -n vss

# MinIO credentials
oc create secret generic minio-creds-secret \
  --from-literal=access-key=minioadmin \
  --from-literal=secret-key=minioadmin \
  -n vss

# Neo4j credentials
oc create secret generic graph-db-creds-secret \
  --from-literal=username=neo4j \
  --from-literal=password=password \
  -n vss
```

Then set `openshift.secrets.create=false` in your values.

### 3. Configure deployment

Export the LLM model. All other configuration (GPU counts, tolerations, security contexts) is defined in [`values-openshift.yaml`](values-openshift.yaml).

```bash
# LLM model — must match nim-llm.model.name in values-openshift.yaml
export LLM_MODEL="meta/llama-3.1-8b-instruct"
```

> **Important:** GPU pods will stay `Pending` if your cluster's GPU node taints don't match the tolerations in `values-openshift.yaml`. The default key is `nvidia.com/gpu`. Find your taint keys and update the `gpuTolerations` anchor in `values-openshift.yaml` before installing:
> ```bash
> oc get nodes -l nvidia.com/gpu.present=true -o name | \
>   xargs -I{} oc describe {} | grep -A1 Taints
> ```

> **Llama 70B:** Set `LLM_MODEL=meta/llama-3.1-70b-instruct` and edit `values-openshift.yaml`: `nim-llm.model.name`, `nim-llm.image.repository`, `global.ucfGlobalEnv[0].value`, `nim-llm.resources.limits.nvidia.com/gpu=4`, `vss.resources.limits.nvidia.com/gpu=2`.

### 4. Install the Helm chart

```bash
helm upgrade --install vss deploy/helm/nvidia-blueprint-vss-2.4.1.tgz \
  -n vss \
  -f deploy/helm/values-openshift.yaml \
  --set openshift.secrets.ngcApiKey="$NGC_API_KEY" \
  --set openshift.secrets.hfToken="$HF_TOKEN"
```

> **If guardrails are enabled** (`DISABLE_GUARDRAILS: "false"` in `values-openshift.yaml`), add the following `--set` flags to override the guardrails model (chart defaults to 70B):
> ```bash
> --set "vss.configs.guardrails_config\.yaml.models[0].engine=nim" \
> --set-string "vss.configs.guardrails_config\.yaml.models[0].model=$LLM_MODEL" \
> --set "vss.configs.guardrails_config\.yaml.models[0].parameters.base_url=http://llm-nim-svc:8000/v1" \
> --set "vss.configs.guardrails_config\.yaml.models[0].type=main"
> ```

The Helm chart will:

1. Create the `vss-sa` service account
2. Create a RoleBinding granting the `anyuid` SCC to `vss-sa`
3. Create all required secrets (NGC registry, NGC API key, HF token, ArangoDB, MinIO, Neo4j)
4. Create an OpenShift Route for external UI access
5. Deploy all 11 pods with OpenShift-compatible security contexts, tolerations, and storage

### 5. Verify the deployment

Check that all pods are running:

```bash
oc get pods -n vss
```

All pods should reach `Running 1/1`. GPU pods (`nim-llm`, `nemo-embedding`, `nemo-rerank`, `vss`) may take 20-30 minutes on first deploy while model weights are downloaded.

**Expected pods:**

| Pod | Purpose | GPU |
|-----|---------|-----|
| `vss-vss-deployment` | Core pipeline + VLM (Cosmos-Reason2-8B) | Yes |
| `nim-llm` | LLM inference (Llama 8B/70B) | Yes |
| `nemo-embedding` | Vector embedding generation | Yes |
| `nemo-rerank` | Document reranking | Yes |
| `milvus-milvus-deployment` | Vector database | No |
| `milvus-minio-*` | Object storage for Milvus | No |
| `etcd-*` | Milvus metadata store | No |
| `elasticsearch-*` | Search engine | No |
| `arango-db-*` | Graph database | No |
| `neo-4-j-*` | Graph database | No |
| `minio-*` | Object storage | No |

Verify the OpenShift resources were created by the Helm template:

```bash
# Route for external access — the UI URL is in the HOST/PORT column
oc get route vss-ui -n vss

# RoleBinding granting anyuid SCC to vss-sa
oc get rolebinding vss-anyuid-scc -n vss

# ServiceAccount
oc get serviceaccount vss-sa -n vss

# Secrets
oc get secrets -n vss | grep -E "ngc-|hf-|arango-|minio-|graph-"
```

To follow progress on specific pods:

```bash
oc logs -f deployment/vss-vss-deployment -n vss
oc logs -f statefulset/nim-llm -n vss
```

### 6. Access the UI

The Helm chart creates an OpenShift Route with TLS edge termination. Get the URL:

```bash
oc get route vss-ui -n vss -o jsonpath='{.spec.host}'
```

Open `https://<route-host>` in a browser.

### 7. Uninstall

```bash
helm uninstall vss -n vss
oc delete pvc --all -n vss
oc delete project vss
```

---

## Model Size Optimization

In GPU-constrained environments, the upstream chart's 70B LLM (4 GPUs) and 2-GPU VLM defaults leave multiple pods `Pending`. The `values-openshift.yaml` overrides these to `llama-3.1-8b-instruct` (1 GPU) and 1 GPU for the VLM respectively.

Configuration in `values-openshift.yaml`:

1. **LLM model and GPU count** — `nim-llm.model.name`, `nim-llm.image.repository`, `nim-llm.resources`, and `global.ucfGlobalEnv[0].value` (LLM_MODEL). Defaults to `meta/llama-3.1-8b-instruct` with 1 GPU.
2. **VLM GPU count** — `vss.resources.limits.nvidia.com/gpu` overrides the default 2-GPU request. The quantized `int4_awq` model fits on a single GPU.

Changing the LLM model also requires updating the model name in multiple locations - see [Challenge 10](#10-llm-model-name-consistency) for details.

---

## OpenShift-Specific Challenges and Solutions

The upstream VSS Helm chart targets vanilla Kubernetes. Running it on OpenShift requires addressing incompatibilities across security contexts, storage permissions, secrets, GPU scheduling, and service configuration. All fixes are applied at install time via `values-openshift.yaml` and `templates/openshift.yaml` (inside the chart) - no post-deploy patching is required.

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

**Solution:** The Helm chart creates a dedicated `vss-sa` service account and a RoleBinding granting the `anyuid` SCC exclusively to it (`templates/openshift.yaml`), scoping the elevated permission to a single named identity rather than the namespace-wide `default` service account.

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

**Solution:** Tolerations are defined in `values-openshift.yaml` for all four GPU services. The default key is `nvidia.com/gpu` — update to match your cluster's GPU node taint keys.

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

**Solution:** The Helm chart template (`templates/openshift.yaml`) creates all required secrets when `openshift.enabled` and `openshift.secrets.create` are set to `true`. API keys are passed via `--set` at install time; service credentials use defaults from `values-openshift.yaml`.

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

**Solution:** The Helm chart creates `hf-token-secret` from the HF token passed via `--set openshift.secrets.hfToken`. The chart already references `hf-token-secret` as an optional `secretKeyRef` - the secret being absent is what caused the silent failure.

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

**Solution:** `DISABLE_GUARDRAILS` is set to `"true"` by default in `values-openshift.yaml` for the 8B configuration. This does not affect core search, summarization, or alert functionality.

```yaml
global:
  ucfGlobalEnv:
  - name: LLM_MODEL
    value: meta/llama-3.1-8b-instruct
  - name: DISABLE_GUARDRAILS
    value: "true"
```

---

### 10. LLM Model Name Consistency

The LLM model name is hardcoded in three independent locations within the chart. Switching the LLM (e.g. from 70B to 8B) without updating all three causes the vss context manager to return 404 errors and guardrails to fall back to NVIDIA's cloud API with 401 Unauthorized.

**Affected Locations:**

- `nim-llm.model.name` - the model identity used by the NIM server itself
- `LLM_MODEL` env var in vss - used by the context manager for chat, summarization, and notifications. Set via `global.ucfGlobalEnv` in `values-openshift.yaml`
- `guardrails_config.yaml` `models[0]` - used by NeMo Guardrails for its startup validation test (only relevant when guardrails are enabled)

**Solution:** All three are configured in `values-openshift.yaml`: `nim-llm.model.name` and `nim-llm.image.repository` for the NIM server, `global.ucfGlobalEnv[0].value` for the LLM_MODEL env var. When guardrails are enabled, the guardrails config model defaults to `meta/llama-3.1-70b-instruct` from the chart — update it if using a different model.

---

## Deployment Files

All OpenShift customizations are codified in the `deploy/helm/` directory alongside the upstream chart:

- **`values-openshift.yaml`** - Helm values overlay for OpenShift. Contains all OpenShift-specific overrides (security contexts, tolerations, secrets, model config, storage).
- **`templates/openshift.yaml`** (inside the chart) - Helm template gated by `openshift.enabled`. Creates ServiceAccount, RoleBinding (anyuid SCC), Route, and secrets.
- **`nvidia-blueprint-vss-2.4.1.tgz`** - The packaged upstream Helm chart with the `openshift.enabled` flag added to `values.yaml`.
