# RetinaSense AI — Frontend

React 18 SPA (Vite + TypeScript + Tailwind) for the RetinaSense AI platform.

## Requirements
- Node.js 22+ and pnpm 11+

## Setup

```bash
pnpm install
pnpm run dev
```

Starts the dev server at `http://localhost:5173`. The SPA calls the backend API at `http://127.0.0.1:8000/api` — start the backend first (see the root README).

## Production build

```bash
pnpm run build
```

Outputs to `dist/` (served by nginx in the Docker image). The API base URL is hardcoded in `src/services/api.ts`.

## Tests

```bash
pnpm run test
```

Vitest suite (9 tests) covering the API client and React hooks.
