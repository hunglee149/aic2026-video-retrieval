param(
    [Parameter(Mandatory = $true)]
    [string]$Image,

    [string]$ResourceGroup = "aic-free-rg",
    [string]$Location = "southeastasia",
    [string]$EnvironmentName = "aic-free-env",
    [string]$AppName = "aic-video-search",
    [ValidateSet("0", "1")]
    [string]$EnableBm25 = "1"
)

$ErrorActionPreference = "Stop"

function Run-Az {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)
    Write-Host "az $($Args -join ' ')" -ForegroundColor Cyan
    & az @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Azure CLI command failed: az $($Args -join ' ')"
    }
}

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw "Azure CLI (az) is not installed. Install it first, then reopen PowerShell."
}

Run-Az extension add --name containerapp --upgrade --only-show-errors
Run-Az provider register --namespace Microsoft.App --wait
Run-Az provider register --namespace Microsoft.OperationalInsights --wait

Run-Az group create `
    --name $ResourceGroup `
    --location $Location `
    --output none

$envExists = (& az containerapp env show --name $EnvironmentName --resource-group $ResourceGroup --query name -o tsv 2>$null)
if (-not $envExists) {
    # WorkloadProfiles gives access to the Consumption profile up to 4 vCPU/8 GiB.
    # logs-destination=none avoids creating a paid Log Analytics workspace.
    Run-Az containerapp env create `
        --name $EnvironmentName `
        --resource-group $ResourceGroup `
        --location $Location `
        --environment-mode WorkloadProfiles `
        --logs-destination none `
        --output none
}

$appExists = (& az containerapp show --name $AppName --resource-group $ResourceGroup --query name -o tsv 2>$null)

$envVars = @(
    "AIC_USE_DUMMY=0",
    "AIC_USE_CLOUD_MEDIA=1",
    "AIC_HF_DATASET_URL=https://huggingface.co/datasets/manhha2502/fullhd/resolve/main",
    "AIC_INDEX_PATH=/app/local/clip_faiss.index",
    "AIC_META_PATH=/app/local/clip_metadata.json",
    "AIC_TEXT_INDEX_PATH=/app/data/input/input/index/text_search_index.pkl",
    "AIC_CLIP_DEVICE=cpu",
    "AIC_TRANSLATION_DEVICE=cpu",
    "AIC_ENABLE_BM25=$EnableBm25",
    "AIC_DISABLE_NEURAL=0"
)

if (-not $appExists) {
    Run-Az containerapp create `
        --name $AppName `
        --resource-group $ResourceGroup `
        --environment $EnvironmentName `
        --workload-profile-name Consumption `
        --image $Image `
        --ingress external `
        --target-port 8000 `
        --transport auto `
        --cpu 4.0 `
        --memory 8.0Gi `
        --min-replicas 0 `
        --max-replicas 1 `
        --env-vars @envVars `
        --output none
} else {
    Run-Az containerapp update `
        --name $AppName `
        --resource-group $ResourceGroup `
        --image $Image `
        --cpu 4.0 `
        --memory 8.0Gi `
        --min-replicas 0 `
        --max-replicas 1 `
        --set-env-vars @envVars `
        --output none
}

$fqdn = (& az containerapp show --name $AppName --resource-group $ResourceGroup --query properties.configuration.ingress.fqdn -o tsv)
Write-Host ""
Write-Host "Deployment complete." -ForegroundColor Green
Write-Host "UI:     https://$fqdn" -ForegroundColor Green
Write-Host "Health: https://$fqdn/healthz" -ForegroundColor Green
Write-Host "Status: https://$fqdn/api/status" -ForegroundColor Green
