#!/bin/bash
# VSS OpenShift Deployment Script
#
# Usage:
#   NGC_API_KEY=your-key HF_TOKEN=hf_... NAMESPACE=vss-blueprint ./deploy-openshift.sh
#   NGC_API_KEY=your-key HF_TOKEN=hf_... NAMESPACE=vss-blueprint LLM_MODEL=meta/llama-3.1-70b-instruct ./deploy-openshift.sh
#   NGC_API_KEY=your-key HF_TOKEN=hf_... NAMESPACE=vss-blueprint GPU_TOLERATION_KEYS=g6-gpu,p4-gpu ./deploy-openshift.sh
#   NGC_API_KEY=your-key HF_TOKEN=hf_... NAMESPACE=vss-blueprint VLM_GPU_COUNT=2 ./deploy-openshift.sh
set -euo pipefail

: "${NGC_API_KEY:?Error: NGC_API_KEY is required}"
: "${NAMESPACE:?Error: NAMESPACE is required}"
: "${HF_TOKEN:?Error: HF_TOKEN is required (accept license at https://huggingface.co/nvidia/Cosmos-Reason2-8B)}"

# Configurable settings
LLM_MODEL="${LLM_MODEL:-meta/llama-3.1-8b-instruct}"
LLM_IMAGE="${LLM_IMAGE:-nvcr.io/nim/meta/llama-3.1-8b-instruct}"
LLM_IMAGE_TAG="${LLM_IMAGE_TAG:-latest}"
LLM_GPU_COUNT="${LLM_GPU_COUNT:-1}"
VLM_GPU_COUNT="${VLM_GPU_COUNT:-1}"

# Comma-separated taint keys on GPU nodes (e.g. "g6-gpu,p4-gpu")
GPU_TOLERATION_KEYS="${GPU_TOLERATION_KEYS:-nvidia.com/gpu}"
GPU_TOLERATION_EFFECT="${GPU_TOLERATION_EFFECT:-NoSchedule}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHART_TGZ="$SCRIPT_DIR/nvidia-blueprint-vss-2.4.1.tgz"

# Create namespace
oc get namespace "$NAMESPACE" &>/dev/null || oc create namespace "$NAMESPACE"

# Create dedicated service account and grant anyuid SCC
# arango-db, neo4j, vss use hardcoded UIDs outside OpenShift's UID range.
oc get serviceaccount vss-sa -n "$NAMESPACE" &>/dev/null || \
  oc create serviceaccount vss-sa -n "$NAMESPACE"
oc adm policy add-scc-to-user anyuid -z vss-sa -n "$NAMESPACE"

# Create NGC registry secret
oc get secret ngc-docker-reg-secret -n "$NAMESPACE" &>/dev/null || \
  oc create secret docker-registry ngc-docker-reg-secret \
    --docker-server=nvcr.io \
    --docker-username='$oauthtoken' \
    --docker-password="$NGC_API_KEY" \
    -n "$NAMESPACE"

# Create required secrets
oc get secret arango-db-creds-secret -n "$NAMESPACE" &>/dev/null || \
  oc create secret generic arango-db-creds-secret \
    --from-literal=username="root" \
    --from-literal=password="password" \
    -n "$NAMESPACE"

oc get secret minio-creds-secret -n "$NAMESPACE" &>/dev/null || \
  oc create secret generic minio-creds-secret \
    --from-literal=access-key="minioadmin" \
    --from-literal=secret-key="minioadmin" \
    -n "$NAMESPACE"

# ngc-api-key-secret — hardcoded secretKeyRef name in nemo-embedding and nemo-rerank.
oc get secret ngc-api-key-secret -n "$NAMESPACE" &>/dev/null || \
  oc create secret generic ngc-api-key-secret \
    --from-literal=NGC_API_KEY="$NGC_API_KEY" \
    -n "$NAMESPACE"

# hf-token-secret — for gated HuggingFace models (e.g. Cosmos-Reason2-8B).
oc get secret hf-token-secret -n "$NAMESPACE" &>/dev/null || \
  oc create secret generic hf-token-secret \
    --from-literal=HF_TOKEN="$HF_TOKEN" \
    -n "$NAMESPACE"

# graph-db-creds-secret — neo4j reads credentials from files mounted by extraPodVolumes.
oc get secret graph-db-creds-secret -n "$NAMESPACE" &>/dev/null || \
  oc create secret generic graph-db-creds-secret \
    --from-literal=username="neo4j" \
    --from-literal=password="password" \
    -n "$NAMESPACE"

# Build GPU toleration args for nim-llm, nemo-embedding, nemo-rerank, vss.
TOLERATION_ARGS=()
IFS=',' read -ra TKEYS <<< "$GPU_TOLERATION_KEYS"
for i in "${!TKEYS[@]}"; do
  key="${TKEYS[$i]}"
  for svc in nim-llm nemo-embedding nemo-rerank vss; do
    TOLERATION_ARGS+=(
      --set "${svc}.tolerations[${i}].key=${key}"
      --set "${svc}.tolerations[${i}].effect=${GPU_TOLERATION_EFFECT}"
      --set "${svc}.tolerations[${i}].operator=Exists"
    )
  done
done

# Optional: disable guardrails (workaround for false positives on multi-image input)
GUARDRAILS_ARGS=()
if [ "${DISABLE_GUARDRAILS:-false}" = "true" ]; then
  GUARDRAILS_ARGS=(
    --set "global.ucfGlobalEnv[1].name=DISABLE_GUARDRAILS"
    --set-string "global.ucfGlobalEnv[1].value=true"
  )
fi

# Deploy
helm upgrade --install vss "$CHART_TGZ" \
  --namespace "$NAMESPACE" \
  -f "$SCRIPT_DIR/values-openshift.yaml" \
  --set "vss.resources.limits.nvidia\.com/gpu=$VLM_GPU_COUNT" \
  --set "nim-llm.image.repository=$LLM_IMAGE" \
  --set "nim-llm.image.tag=$LLM_IMAGE_TAG" \
  --set "nim-llm.resources.limits.nvidia\.com/gpu=$LLM_GPU_COUNT" \
  --set "nim-llm.model.name=$LLM_MODEL" \
  --set "global.ucfGlobalEnv[0].name=LLM_MODEL" \
  --set-string "global.ucfGlobalEnv[0].value=$LLM_MODEL" \
  --set "vss.configs.guardrails_config\.yaml.models[0].engine=nim" \
  --set-string "vss.configs.guardrails_config\.yaml.models[0].model=$LLM_MODEL" \
  --set "vss.configs.guardrails_config\.yaml.models[0].parameters.base_url=http://llm-nim-svc:8000/v1" \
  --set "vss.configs.guardrails_config\.yaml.models[0].type=main" \
  "${TOLERATION_ARGS[@]}" \
  "${GUARDRAILS_ARGS[@]}"

# Wait for rollout (GPU pods may take 20-30 min on first deploy)
echo "Waiting for deployments to be ready..."
for resource in $(oc get deploy,statefulset -n "$NAMESPACE" -o name); do
  echo "  Waiting for ${resource#*/}..."
  oc rollout status "$resource" -n "$NAMESPACE" --timeout=30m || \
    echo "  Warning: ${resource#*/} not ready — check: oc get pods -n $NAMESPACE"
done

# Expose UI
oc expose svc/vss-service --port=9000 --name=vss-ui -n "$NAMESPACE" 2>/dev/null || true

ROUTE=$(oc get route vss-ui -n "$NAMESPACE" -o jsonpath='{.spec.host}' 2>/dev/null || true)
echo ""
echo "=== Done ==="
[ -n "$ROUTE" ] && echo "UI: http://$ROUTE"