#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:?Usage: ./deploy/deploy_azure.sh ghcr.io/USER/aic-video-retrieval:latest [0|1]}"
ENABLE_BM25="${2:-1}"
RG="${RG:-aic-free-rg}"
LOCATION="${LOCATION:-southeastasia}"
ENV_NAME="${ENV_NAME:-aic-free-env}"
APP_NAME="${APP_NAME:-aic-video-search}"

az extension add --name containerapp --upgrade --only-show-errors
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.OperationalInsights --wait
az group create -n "$RG" -l "$LOCATION" -o none

if ! az containerapp env show -n "$ENV_NAME" -g "$RG" >/dev/null 2>&1; then
  az containerapp env create \
    -n "$ENV_NAME" -g "$RG" -l "$LOCATION" \
    --environment-mode WorkloadProfiles \
    --logs-destination none \
    -o none
fi

ENV_VARS=(
  "AIC_USE_DUMMY=0"
  "AIC_USE_CLOUD_MEDIA=1"
  "AIC_HF_DATASET_URL=https://huggingface.co/datasets/manhha2502/fullhd/resolve/main"
  "AIC_INDEX_PATH=/app/local/clip_faiss.index"
  "AIC_META_PATH=/app/local/clip_metadata.json"
  "AIC_TEXT_INDEX_PATH=/app/data/input/input/index/text_search_index.pkl"
  "AIC_CLIP_DEVICE=cpu"
  "AIC_TRANSLATION_DEVICE=cpu"
  "AIC_ENABLE_BM25=$ENABLE_BM25"
  "AIC_DISABLE_NEURAL=0"
)

if az containerapp show -n "$APP_NAME" -g "$RG" >/dev/null 2>&1; then
  az containerapp update \
    -n "$APP_NAME" -g "$RG" \
    --image "$IMAGE" \
    --cpu 4.0 --memory 8.0Gi \
    --min-replicas 0 --max-replicas 1 \
    --set-env-vars "${ENV_VARS[@]}" \
    -o none
else
  az containerapp create \
    -n "$APP_NAME" -g "$RG" \
    --environment "$ENV_NAME" \
    --workload-profile-name Consumption \
    --image "$IMAGE" \
    --ingress external --target-port 8000 --transport auto \
    --cpu 4.0 --memory 8.0Gi \
    --min-replicas 0 --max-replicas 1 \
    --env-vars "${ENV_VARS[@]}" \
    -o none
fi

FQDN="$(az containerapp show -n "$APP_NAME" -g "$RG" --query properties.configuration.ingress.fqdn -o tsv)"
echo "UI:     https://$FQDN"
echo "Health: https://$FQDN/healthz"
echo "Status: https://$FQDN/api/status"
