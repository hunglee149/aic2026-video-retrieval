# Deploy AIC 2026 lên Azure Container Apps gần như miễn phí

Kiến trúc triển khai:

- Azure Container Apps: FastAPI/UI + CLIP text encoder + BM25 + FAISS.
- Hugging Face dataset `manhha2502/fullhd`: giữ video, keyframe và nguồn index.
- GitHub Container Registry (GHCR): lưu Docker image public.
- Không dùng Azure Blob Storage, Azure Files, Azure Container Registry hay Log Analytics để tránh phát sinh chi phí không cần thiết.

## 1. Đẩy source lên GitHub

Đưa toàn bộ repo đã sửa này lên một GitHub repository. Workflow `.github/workflows/build-container.yml` sẽ build Docker image trên GitHub nên máy local không bắt buộc phải có Docker.

## 2. Build image bằng GitHub Actions

Vào GitHub repository:

1. `Actions`
2. `Build AIC container`
3. `Run workflow`
4. Giữ `Bake BM25 text index into the image = true`
5. `Run workflow`

Image tạo ra có dạng:

`ghcr.io/<github-username>/aic-video-retrieval:latest`

Lần build đầu sẽ tải khoảng 363 MB CLIP FAISS, metadata, khoảng 564 MB text index và model weights từ Hugging Face.

## 3. Chuyển GHCR package sang Public

Sau khi workflow build xong:

1. Mở profile GitHub.
2. Vào `Packages`.
3. Chọn `aic-video-retrieval`.
4. `Package settings`.
5. `Change visibility` -> `Public`.

Azure Container Apps có thể pull public GHCR image mà không cần token.

## 4. Cài Azure CLI và đăng nhập

PowerShell:

```powershell
az login
az account show
```

Nếu có nhiều subscription:

```powershell
az account list -o table
az account set --subscription "TEN_SUBSCRIPTION"
```

## 5. Deploy

Trong thư mục repo:

```powershell
powershell -ExecutionPolicy Bypass -File .\deploy\deploy_azure.ps1 `
  -Image "ghcr.io/<github-username>/aic-video-retrieval:latest"
```

Script tạo:

- Resource Group: `aic-free-rg`
- Container Apps Environment: `aic-free-env`
- Region: `southeastasia`
- App: `aic-video-search`
- Consumption profile: `4 vCPU / 8 GiB`
- `min replicas = 0`
- `max replicas = 1`
- logs destination = `none`

Cuối lệnh sẽ in URL UI.

## 6. Kiểm tra

Mở:

- `https://<fqdn>/healthz` -> phải trả `{"ok": true}` nhanh, không load model.
- `https://<fqdn>/api/status` -> request đầu có thể chậm vì app scale từ 0 và load model/index.

Trong `/api/status`, CLIP và BM25 nên có `state = ready`.

Sau đó mở `https://<fqdn>/`, nhập query và search.

## 7. Nếu Azure báo Out Of Memory

Không cần rebuild image. Tắt BM25 để giảm RAM:

```powershell
az containerapp update `
  --name aic-video-search `
  --resource-group aic-free-rg `
  --set-env-vars AIC_ENABLE_BM25=0
```

Khi đó hệ thống vẫn search bằng CLIP + FAISS và vẫn xem video/keyframe từ Hugging Face.

Bật lại:

```powershell
az containerapp update `
  --name aic-video-search `
  --resource-group aic-free-rg `
  --set-env-vars AIC_ENABLE_BM25=1
```

## 8. Xem log realtime mà không lưu Log Analytics

```powershell
az containerapp logs show `
  --name aic-video-search `
  --resource-group aic-free-rg `
  --follow
```

## 9. Update code về sau

1. Push code mới lên GitHub.
2. Vào Actions và chạy lại `Build AIC container`.
3. Sau khi build xong:

```powershell
az containerapp update `
  --name aic-video-search `
  --resource-group aic-free-rg `
  --image "ghcr.io/<github-username>/aic-video-retrieval:latest"
```

Nếu Azure vẫn giữ revision cũ do tag `latest`, dùng tag SHA mà workflow cũng đã push, ví dụ:

`ghcr.io/<github-username>/aic-video-retrieval:<commit-sha>`

## 10. Xóa toàn bộ Azure resource khi không dùng nữa

```powershell
az group delete --name aic-free-rg --yes --no-wait
```

Lệnh này xóa Container App và environment trong resource group.
