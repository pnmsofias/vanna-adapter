# vanna-adapter

CLI tool to query databases using natural language via vanna.ai.

Run `vanna-adapter` after setting the required environment variables:
`LLM_API_KEY`, `DB_URL` and `VANNA_QUERY`.

- `DB_URL` acepta URLs estándar como `postgresql://user:pass@host/db`.
- Si pasas un enlace HTTP(S) que termina en `.sqlite`, el CLI descarga el archivo en `<directorio actual>/databases/` y lo expone como `sqlite:///…` de forma automática (útil para bases como `https://vanna.ai/Chinook.sqlite`). Ese archivo queda cacheado (nombre con hash del URL) y antes de reutilizarlo se valida contra los metadatos HTTP (ETag/Last-Modified/Content-Length); sólo se vuelve a descargar si el servidor indica cambios o no es posible validar. Elimina el archivo para forzar una actualización manual. El valor de `DB_URL` también se reescribe internamente para que cualquier componente que lo lea vea la ruta local.
