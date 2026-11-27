# Kubernetes

```bash
kubectl apply -f deploy/k8s/deployment.yaml
kubectl apply -f deploy/k8s/service.yaml
kubectl get pods -l app=draftfly
```

清单见 `deploy/k8s/`（Deployment / Service / ConfigMap / HPA）。
