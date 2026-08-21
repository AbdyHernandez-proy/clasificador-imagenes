# ML

Area para datasets, entrenamiento, modelos propios e inferencia.

## Estructura

```text
ml/
|-- datasets/
|-- models/
|-- training/
|-- inference/
`-- registry.json
```

## Registro de modelos

`registry.json` reemplaza una base de datos inicial. El backend lo lee para saber que modelos existen y cual usar por defecto.

Ejemplo futuro:

```json
{
  "default_model": "frutas-v1",
  "models": [
    {
      "id": "frutas-v1",
      "name": "Clasificador de frutas",
      "version": "v1",
      "path": "ml/models/frutas/v1/model.keras",
      "labels": "ml/models/frutas/v1/labels.json",
      "metrics": "ml/models/frutas/v1/metrics.json"
    }
  ]
}
```
