{{- define "draftfly.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "draftfly.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "draftfly.name" .) | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "draftfly.labels" -}}
app.kubernetes.io/name: {{ include "draftfly.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}
