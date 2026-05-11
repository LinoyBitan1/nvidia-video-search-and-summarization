# Deploying NVIDIA VSS on Red Hat OpenShift AI

## What We're Deploying

The VSS Blueprint runs a video analytics pipeline that ingests video (file upload or RTSP live stream), captions individual frames using a Vision Language Model, and makes the content searchable via natural language.

| Component | Image | GPU | Purpose |
|-----------|-------|-----|---------|
| **vss** | NVCF chart | 1 | Core pipeline + VLM (Cosmos-Reason2-8B) |
| **nim-llm** (NIM Operator) | `nvcr.io/nim/meta/llama-3.1-8b-instruct` | 1 | LLM inference via NIMService |
| **nemo-embedding** (NIM Operator) | `nvcr.io/nim/nvidia/llama-3.2-nv-embedqa-1b-v2` | 1 | Vector embedding via NIMService |
| **nemo-rerank** (NIM Operator) | `nvcr.io/nim/nvidia/llama-3.2-nv-rerankqa-1b-v2` | 1 | Document reranking via NIMService |
| **milvus** | NVCF subchart | 0 | Vector database |
| **elasticsearch** | NVCF subchart | 0 | Search engine |
| **arango-db** | NVCF subchart | 0 | Graph database |
| **neo4j** | NVCF subchart | 0 | Graph database |
| **minio** / **milvus-minio** | NVCF subchart | 0 | Object storage |
| **etcd** | NVCF subchart | 0 | Milvus metadata store |

**Data flow:**

- **Upload:** video → vss (Cosmos captions each frame) → nemo-embedding → Milvus
- **Search:** user query → nemo-embedding → Milvus (vector search) → nemo-rerank → nim-llm → response
- **Alerts:** vss monitors captions for user-defined event keywords

**Total:** 4 GPUs minimum (1 per GPU pod).

### NIM Operator Components

When deployed with the NIM Operator (recommended on OpenShift), each NIM inference
service is managed as a pair of custom resources:

- **NIMCache** — downloads and caches the model on a PVC. Annotated with
  `helm.sh/resource-policy: keep` to survive upgrades and uninstalls.
- **NIMService** — runs the inference server, references its NIMCache for model storage,
  manages replicas, GPU allocation, and health probes.

## Tested Hardware

| Parameter | Value |
|-----------|-------|
| Platform | Red Hat OpenShift AI (RHOAI) 4.14+ |
| GPU nodes | 1+ nodes with NVIDIA L40S / A100 40GB / H100 |
| GPUs per node | 1+ |
| Total GPUs | 4 (one per GPU pod) |
| VRAM | 40 GB per GPU minimum |
| CPU | 8+ cores across GPU nodes |
| RAM | 64 GB+ per GPU node |
| Storage | 310 GiB dynamically provisioned PVCs (100 GiB LLM + 50 GiB embedding + 50 GiB rerank + 100 GiB VLM + 10 GiB etcd) |
| API keys | [NGC API key](https://org.ngc.nvidia.com/setup/api-keys) + [HuggingFace token](https://huggingface.co/nvidia/Cosmos-Reason2-8B) (gated Cosmos-Reason2-8B license accepted) |

Minimum for reproduction: 4 × NVIDIA L40S / A100 / H100 GPUs (40 GB+ VRAM each), 310 GiB storage. A10G (22 GB) is **not sufficient** — the VLM (Cosmos-Reason2-8B) requires ~22 GiB for model weights + KV cache.

Validated on OpenShift 4.19 on AWS with `g6e.2xlarge` (1× L40S 46 GB) and `p4d.24xlarge` (8× A100 40GB) GPU nodes.

## What's Different from Upstream

| Area | Upstream Default | OpenShift Deployment | Impact |
|------|------------------|----------------------|--------|
| Pod management | Subchart Deployments | NIM Operator (NIMCache + NIMService) | Automated model download, cache lifecycle |
| External access | `kubectl port-forward` | OpenShift Route with TLS | Production-grade ingress |
| Volume provisioning | `hostPath` volumes | Dynamic PVCs via NIM Operator | No node-local dependency |
| Security context | Pods require root / specific UIDs | Custom SCC + RoleBinding | Compatible with OpenShift SCC |
| Secrets | Manual creation | Helm-managed via `nvcf` + `openshift.secrets` | Single `helm install` creates all secrets |
| LLM model | `llama-3.1-70b-instruct` (4 GPU) | `llama-3.1-8b-instruct` (1 GPU) | Reduced GPU footprint |
| VLM GPU count | 2 GPUs | 1 GPU (quantized `int4_awq`) | Reduced GPU footprint |

> **Running with the upstream 70B LLM:** Requires 8 GPUs minimum (4 for LLM tensor
> parallelism on a single node + 2 for VLM + 1 embedding + 1 rerank) with NVIDIA A100
> 80GB or higher. Update `values-openshift.yaml` accordingly.

## Deployment Files

All OpenShift customizations are codified in the `deploy/helm/` directory alongside the upstream chart:

- **`values-openshift.yaml`** — Helm values overlay for OpenShift. Contains all OpenShift-specific overrides (security contexts, tolerations, secrets, model config, storage).
- **`templates/openshift.yaml`** (inside the chart) — Helm template gated by `openshift.enabled`. Creates custom SCC, RoleBinding, Route, and secrets.
- **`templates/nim-llm.yaml`**, **`nim-embedding.yaml`**, **`nim-reranking.yaml`** (inside the chart) — NIMCache + NIMService templates for each NIM, gated by the NIM Operator CRD.
- **`nvidia-blueprint-vss-2.4.1.tgz`** — The packaged upstream Helm chart with the `openshift.enabled` flag and NIM Operator support added to `values.yaml`.

## Prerequisites

### CLI Tools

- `oc` (OpenShift CLI) logged into your cluster
- `helm` v3.12+

### Cluster Requirements

- Red Hat OpenShift 4.14+
- NVIDIA GPU Operator installed and configured
- NIM Operator installed (provides `apps.nvidia.com/v1alpha1` API)
- At least 4 GPU nodes available

### Verify GPU Availability

```bash
oc get nodes -l nvidia.com/gpu.present=true
oc describe node <gpu-node> | grep -A 5 "Allocatable"
```

## Configuration Reference

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NGC_API_KEY` | Yes | — | NGC API key for image pulls and model downloads |
| `HF_TOKEN` | Yes | — | HuggingFace token for gated Cosmos-Reason2-8B model |

### OpenShift Block (`openshift:`)

| Key | Default | Description |
|-----|---------|-------------|
| `openshift.enabled` | `false` | Master toggle for all OpenShift resources |
| `openshift.route.enabled` | `false` | Create an OpenShift Route for the UI |
| `openshift.scc.create` | `false` | Create custom SCC for NIM and subchart pods |
| `openshift.scc.priority` | `10` | SCC priority |
| `openshift.secrets.create` | `false` | Create arango/minio/neo4j credential secrets |

### NIM Operator Block (`nimOperator:`)

Each NIM (`nim-llm`, `nemo-embedding`, `nemo-rerank`) has the same schema:

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `false` | Deploy this NIM via NIM Operator |
| `replicas` | `1` | Number of inference replicas |
| `service.name` | (per NIM) | Kubernetes service name (used for DNS) |
| `image.repository` | (per NIM) | NGC container image |
| `image.tag` | (per NIM) | Image version |
| `resources.limits.nvidia.com/gpu` | `1` | GPU allocation |
| `storage.pvc.size` | `50–100Gi` | Model cache PVC size |
| `storage.pvc.storageClass` | `""` | StorageClass (uses cluster default if empty) |
| `expose.service.port` | `8000` | Service port |
| `tolerations` | `[]` | Node tolerations for GPU taints |
| `env` | `[]` | Environment variables |
| `startupProbe` | (per NIM) | Startup probe configuration |

## Deployment

### 1. Create Namespace

```bash
oc new-project vss
```

### 2. Create Secrets

The chart can create secrets automatically via the `nvcf` mechanism and
`openshift.secrets.create: true` — just pass API keys via `--set` at install time.
Alternatively, create all secrets manually before install (recommended for production).

> **Helm auto-creation:** If using auto-creation, skip this step and pass keys via
> `--set` flags in step 3. If you pre-create secrets here, you **must** disable the
> chart's secret creation to prevent it from overwriting your secrets with empty
> defaults. In `values-openshift.yaml`, set:
> ```yaml
> nvcf:
>   dockerRegSecrets: []
>   additionalSecrets: []
>
> openshift:
>   secrets:
>     create: false
> ```
> Also omit the `--set` flags in step 3.

```bash
export NGC_API_KEY="<your-ngc-api-key>"
export HF_TOKEN="<your-huggingface-token>"
export NAMESPACE=vss

# Image pull secret (for pulling NIM containers from nvcr.io)
oc create secret docker-registry ngc-docker-reg-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="${NGC_API_KEY}" \
  -n $NAMESPACE

# Label for Helm adoption
oc label secret ngc-docker-reg-secret \
  app.kubernetes.io/managed-by=Helm -n $NAMESPACE
oc annotate secret ngc-docker-reg-secret \
  meta.helm.sh/release-name=vss \
  meta.helm.sh/release-namespace=$NAMESPACE -n $NAMESPACE
```

Create the NGC API secret for the NIM Operator:

```bash
oc create secret generic ngc-api-key-secret \
  --from-literal=NGC_API_KEY="${NGC_API_KEY}" \
  -n $NAMESPACE

oc label secret ngc-api-key-secret \
  app.kubernetes.io/managed-by=Helm -n $NAMESPACE
oc annotate secret ngc-api-key-secret \
  meta.helm.sh/release-name=vss \
  meta.helm.sh/release-namespace=$NAMESPACE -n $NAMESPACE
```

Create the HuggingFace token secret for the gated Cosmos-Reason2-8B model:

```bash
oc create secret generic hf-token-secret \
  --from-literal=HF_TOKEN="${HF_TOKEN}" \
  -n $NAMESPACE

oc label secret hf-token-secret \
  app.kubernetes.io/managed-by=Helm -n $NAMESPACE
oc annotate secret hf-token-secret \
  meta.helm.sh/release-name=vss \
  meta.helm.sh/release-namespace=$NAMESPACE -n $NAMESPACE
```

Create the service credential secrets:

```bash
# ArangoDB credentials
oc create secret generic arango-db-creds-secret \
  --from-literal=username=root \
  --from-literal=password="<your-arango-password>" \
  -n $NAMESPACE

# MinIO credentials
oc create secret generic minio-creds-secret \
  --from-literal=access-key="<your-minio-access-key>" \
  --from-literal=secret-key="<your-minio-secret-key>" \
  -n $NAMESPACE

# Neo4j credentials
oc create secret generic graph-db-creds-secret \
  --from-literal=username=neo4j \
  --from-literal=password="<your-neo4j-password>" \
  -n $NAMESPACE

# Label all service secrets for Helm adoption
for secret in arango-db-creds-secret minio-creds-secret graph-db-creds-secret; do
  oc label secret $secret app.kubernetes.io/managed-by=Helm -n $NAMESPACE
  oc annotate secret $secret meta.helm.sh/release-name=vss \
    meta.helm.sh/release-namespace=$NAMESPACE -n $NAMESPACE
done
```

Link the pull secret to the NIM Operator service account:

```bash
oc create sa nim-cache-sa -n $NAMESPACE || true
oc secrets link nim-cache-sa ngc-docker-reg-secret --for=pull -n $NAMESPACE
```

### 3. Install the Chart

```bash
helm upgrade --install vss deploy/helm/nvidia-blueprint-vss-2.4.1.tgz \
  -n $NAMESPACE \
  -f deploy/helm/values-openshift.yaml \
  --set nvcf.dockerRegSecrets[0].password="$NGC_API_KEY" \
  --set nvcf.additionalSecrets[0].stringData.value="$NGC_API_KEY" \
  --set nvcf.additionalSecrets[1].stringData.value="$HF_TOKEN"
```

> If you pre-created secrets in step 2, omit the `--set` flags above.

> **If you changed the LLM model** (e.g. from 70B to 8B), add the following `--set` flags to override the guardrails model accordingly:
> ```bash
> --set "vss.configs.guardrails_config\.yaml.models[0].engine=nim" \
> --set-string "vss.configs.guardrails_config\.yaml.models[0].model=$LLM_MODEL" \
> --set "vss.configs.guardrails_config\.yaml.models[0].parameters.base_url=http://llm-nim-svc:8000/v1" \
> --set "vss.configs.guardrails_config\.yaml.models[0].type=main"
> ```

This creates:
- 3 × NIMCache (model download and caching)
- 3 × NIMService (inference servers)
- 1 × OpenShift Route (external UI access)
- 1 × Custom SCC (`vss-nim`)
- 1 × RoleBinding (binds SCC to `default`, `nim-cache-sa`, and NIMService SAs)
- All required secrets (NGC registry, NGC API key, HF token, ArangoDB, MinIO, Neo4j)
- VSS application and supporting services (Milvus, Neo4j, ArangoDB, Elasticsearch, etc.)
- Subchart NIM Deployments are disabled (`nim-llm.enabled: false`, etc.)

### 4. Monitor Model Downloads

NIMCache resources download models from NGC. This can take 10–60 minutes depending
on model size and network speed.

```bash
oc get nimcache -n $NAMESPACE -w
```

Wait until all caches show `Ready`:

```
NAME                                                          STATUS   AGE
llm-nim-svc-cache                                             Ready    30m
nemo-embedding-embedding-deployment-embedding-service-cache   Ready    15m
nemo-rerank-ranking-deployment-ranking-service-cache          Ready    15m
```

## Verification

### Check NIMService Status

```bash
oc get nimservice -n $NAMESPACE
```

### Check Pods

```bash
oc get pods -n $NAMESPACE
```

All pods should reach `Running 1/1`. The VSS pod has init containers that wait for Milvus, Neo4j, and the LLM to become available.

### Health Endpoints

```bash
for svc in llm-nim-svc nemo-embedding-embedding-deployment-embedding-service nemo-rerank-ranking-deployment-ranking-service; do
  echo -n "$svc: "
  oc exec -n $NAMESPACE deployment/$svc -- curl -s http://localhost:8000/v1/health/ready
  echo
done
```

## Accessing the UI

The Helm chart creates an OpenShift Route with TLS edge termination. Get the URL:

```bash
oc get route vss-ui -n $NAMESPACE -o jsonpath='{.spec.host}'
```

Open `https://<route-host>` in a browser.

## OpenShift-Specific Challenges and Solutions

### 1. Security Context Constraints

**What:** Several containers (arango-db, milvus, milvus-minio) run as root by default. OpenShift's `restricted-v2` SCC blocks this by forcing a random UID.

**Error:** `container has runAsNonRoot and image has non-numeric user`

**Services affected:** arango-db, milvus, milvus-minio.

**Fix:** The custom SCC (`<release>-nim`) sets `runAsUser: RunAsAny`, which allows containers to run as the UID defined in their image. The SCC is created declaratively by the Helm chart when `openshift.scc.create: true`.

### 2. hostPath Not Available on OpenShift

**What:** The upstream chart defaults to `hostPath` volumes for NIM model storage. OpenShift restricts `hostPath` to privileged pods.

**Fix:** The NIM Operator path creates dynamically provisioned PVCs via NIMCache instead. Set `storage.pvc.storageClass` to your cluster's StorageClass (e.g. `gp3-csi`). The subchart `hostPath` volumes are unused when `nim-llm.enabled: false`.

### 3. GPU Node Tolerations

**What:** GPU nodes carry `NoSchedule` taints. Without matching tolerations, pods stay `Pending`.

**Error:** `0/N nodes are available: N node(s) had untolerated taint`

**Services affected:** vss, nim-llm, nemo-embedding, nemo-rerank (all GPU pods).

**Fix:** Add tolerations in the values overlay for each GPU service:

```yaml
nimOperator:
  nim-llm:
    tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
```

Check your taints with `oc describe node <gpu-node> | grep Taints`.

### 4. Missing Secrets

**What:** The chart references secrets that don't exist by default. Pods fail with `secret not found`. The HF_TOKEN secret is especially easy to miss — without it the VLM (Cosmos-Reason2-8B) download fails silently and the server never starts.

**Error:** `secret "<name>" not found`

**Services affected:** All (shared secrets).

**Fix:** NGC and HF secrets are created via the `nvcf` mechanism. Service credential secrets (ArangoDB, MinIO, Neo4j) are created by `openshift.yaml` when `openshift.secrets.create: true`.

### 5. TOKENIZERS_PARALLELISM Race Condition

**What:** The HuggingFace tokenizers library has a thread pool race condition that can cause NIMs to crash or fail startup probes intermittently.

**Services affected:** nim-llm, nemo-embedding, nemo-rerank.

**Fix:** All NIM Operator env blocks include `TOKENIZERS_PARALLELISM=false` as a preventive measure.

### 6. Helm Secret Adoption

**What:** Secrets created before `helm install` cause adoption errors during installation or upgrades.

**Error:** `"Error: rendered manifests contain a resource that already exists"`

**Services affected:** All (shared secrets).

**Fix:** Label secrets with Helm metadata before install:

```bash
oc label secret <name> app.kubernetes.io/managed-by=Helm
oc annotate secret <name> meta.helm.sh/release-name=vss meta.helm.sh/release-namespace=$NAMESPACE
```

### 7. NIMCache PVC Sizing

**What:** NIM model cache PVCs must be large enough to hold all downloaded model profiles. Undersized PVCs cause download failures.

**Services affected:** nim-llm, nemo-embedding, nemo-rerank.

**Fix:** PVC sizes in `values-openshift.yaml`: LLM 100 GiB, embedding 50 GiB, reranking 50 GiB. NIMCache PVCs are immutable — delete the NIMCache and PVC to resize, then re-run `helm install`.

## Cleanup

```bash
# Uninstall the Helm release
helm uninstall vss -n $NAMESPACE

# NIMCache PVCs persist by default (helm.sh/resource-policy: keep).
# Delete manually if you want to reclaim storage:
oc delete nimcache --all -n $NAMESPACE
oc delete pvc -l app.nvidia.com/nim-cache -n $NAMESPACE

# Delete the project
oc delete project $NAMESPACE
```
